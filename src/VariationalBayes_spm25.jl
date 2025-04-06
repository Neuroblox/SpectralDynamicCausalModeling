using ModelingToolkit
using OrderedCollections

# compute Jacobian of rhs w.r.t. variable -> matrix exponential solution (use ExponentialUtilities.jl)
# -> use this numerical integration as solution to the diffeq to then differentiate solution w.r.t. parameters (like sensitivity analysis in Ma et al. 2021)
# -> that Jacobian is used in all the computations of the variational Bayes

"""
    This is K(ω) in the spectral DCM paper.
"""
function transferfunction_fmri(x, w, μθ_pr, C, lnϵ, lndecay, lntransit)   # relates to: spm_dcm_mtf.m
    # compute transfer function of Volterra kernels, see fig 1 in friston2014 (spectral DCM paper)
    # 1. compute jacobian w.r.t. f ; TODO: what is it with this "delay operator" that is set to 1 in "spm_fx_fmri.m"
    # J_x = jacobian(f, x0) # well, no need to perform this for a linear system... we already have it: μθ_pr
    C /= 16.0   # TODO: unclear why C is devided by 16 but see spm_fx_fmri.m:49
    # 2. get jacobian of hemodynamics
    J = hemodynamics_jacobian(x[:, 2:end], lndecay, lntransit)
    μθ_pr -= diagm(exp.(diag(μθ_pr))/2 + diag(μθ_pr))
    # if I eventually need also the change of variables rather than just the derivative then here is where to fix it!
    nd = size(μθ_pr, 1)
    J_tot = [μθ_pr zeros(nd, size(J, 2));   # add derivatives w.r.t. neural signal; this is the total Jacobian of the underlying dynamics ∂ₓf in the paper
            [Matrix(1.0I, size(μθ_pr)); zeros(size(J)[1]-nd, size(μθ_pr)[2])] J]

    dfdu = [C;
            zeros(size(J,1), size(C, 2))]

    F = eigen(J_tot, sortby=nothing, permute=true)
    Λ = F.values
    V = F.vectors

    # condition unstable eigenmodes
    # if max(w) > 1
    #     s = 1j*imag(s) + real(s) - exp(real(s));
    # else
    #     s = 1j*imag(s) + min(real(s),-1/32);
    # end

    # 3. get jacobian (??) of bold signal, just compute it as is done, but how is this a jacobian, it isn't! if anything it should be a gradient since the BOLD signal is scalar
    #TODO: implement numerical and compare with analytical: J_g = jacobian(bold, x0)
    dgdx = boldsignal(x, lnϵ)[2]
    dgdv = dgdx*V[end-size(dgdx,2)+1:end, :]     # TODO: not a clean solution, also not in the original code since it seems that the code really depends on the ordering of eigenvalues and respectively eigenvectors!
    dvdu = pinv(V)*dfdu

    nw = size(w,1)            # number of frequencies
    ng = size(dgdx,1)         # number of outputs
    nu = size(dfdu,2)         # number of inputs
    nk = size(V,2)            # number of modes
    S = zeros(Complex, nw, ng, nu)

    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*w .- Λ[k]).^-1    # TODO: clean up 1im*2*pi*freq instead of omega to be consistent with the usual nomenclature
                S[:,i,j] .+= dgdv[i,k]*dvdu[k,j]*Sk
            end
        end
    end
    return S
end

"""
    Main function in which actually some interesting computation happens. This function implements equation 2 of the spectral DCM paper.
    Note that nomenclature is taken from SPM12 code and it does not seem to coincide with the spectral DCM paper's nomenclature. 
    For instance, Gu should represent the spectral component due to external input according to the paper. However, in the code this represents
    the hidden state fluctuations (which are called Gν in the paper).
    Gn in the code corresponds to Ge in the paper, i.e. the observation noise. In the code global and local components are defined, no such distinction
    is discussed in the paper. In fact the parameter γ, corresponding to local component is not present in the paper.
"""
function csd_approx(x, w, μθ_pr, C, α::Vector, β, γ, lnϵ, lndecay, lntransit)
    # priors of spectral parameters
    # region specific neuronal noise ln(α), general measurement noise ln(β), region specific measurement noise: ln(γ)
    nw = length(w)
    nd = size(μθ_pr, 1)

    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    Gu = zeros(nw, nd, nd)
    Gn = zeros(nw, nd, nd)
    G = w.^(-exp(β[1]))    # spectrum of hidden dynamics
    G /= sum(G)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1])*G
    end
    # global components and region specific observation noise (1/f or AR(1) form)
    # region specific observation noise (1/f or AR(1) form)
    G = w.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nd
        Gn[:,i,i] .+= exp(γ[i])*G
    end
    for i = 1:nd
        for j = i:nd
            Gn[:,i,j] .+= exp(α[2])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end

    C = Matrix(I, nd, nd)     # here C is overwritten, whatever it was before, doesn't matter. This is SPM12 code, unclear how this makes sense.

    S = transferfunction_fmri(x, w, μθ_pr, C, lnϵ, lndecay, lntransit)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(ComplexF64,nw,nd,nd);
    for i = 1:nw
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end


"""
    Main function that computes the CSD: first arrange parameters, then call csd_approx which actually computes the CSD, and finally transform and back-transform to and from MAR.
    It is unclear why this last step is performed. A possible purpose is smoothing of the CSD, but this step is not documented anywhere and just taken as is from SPM12.
"""
function csd_fmri_mtf(x, freqs, p, param)   # alongside the above realtes to spm_csd_fmri_mtf.m
    dim = size(x, 1)
    μθ_pr = reshape(param[1:dim^2], dim, dim)
    C = param[(1+dim^2):(dim+dim^2)]
    lntransit = param[(1+dim+dim^2):(2dim+dim^2)]
    lndecay = param[1+2dim+dim^2]
    lnϵ = param[2+2dim+dim^2]
    α = param[[3+2dim+dim^2, 5+2dim+dim^2]]
    β = param[[4+2dim+dim^2, 6+2dim+dim^2]]
    γ = param[(7+2dim+dim^2):(6+3dim+dim^2)]
    G = csd_approx(x, freqs, μθ_pr, C, α, β, γ, lnϵ, lndecay, lntransit)
    dt = 1/(2*freqs[end])

    # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
    # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
    # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. Friston conferms that likely it is
    # to make y well behaved.
    mar = csd2mar(G, freqs, dt, p-1)
    y = mar2csd(mar, freqs)
    return y
end

function diff(U, dx, f, param::Vector)
    nJ = size(U, 2)
    y0 = f(param)
    J = zeros(ComplexF64, nJ, size(y0, 1), size(y0, 2), size(y0, 3))
    for i = 1:nJ
        y1 = f(param .+ U[:, i]*dx)
        J[i,:,:,:] = (y1 .- y0)/dx
    end
    return J, y0
end

function diff(U, dx, f, param::OrderedDict)
    nJ = size(U, 2)
    y0 = f(param)
    J = zeros(ComplexF64, nJ, size(y0, 1), size(y0, 2), size(y0, 3))
    for i = 1:nJ
        tmp_param = vecparam(param) .+ U[:, i]*dx
        y1 = f(unvecparam(tmp_param, param))
        J[i,:,:,:] = (y1 .- y0)/dx
    end
    return J, y0
end

function spm_logdet(M)
    TOL = 1e-16
    s = diag(M)
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
end

function csd_Q(csd)
    s = size(csd)
    Qn = length(csd)
    Q = zeros(ComplexF64, Qn, Qn);
    idx = CartesianIndices(csd)
    for Qi  = 1:Qn
        for Qj = 1:Qn
            if idx[Qi][1] == idx[Qj][1]
                Q[Qi,Qj] = csd[idx[Qi][1], idx[Qi][2], idx[Qj][2]]*csd[idx[Qi][1], idx[Qi][3], idx[Qj][3]]
            end
        end
    end
    return inv(Q .+ opnorm(Q, 1)/32*Matrix(I, size(Q)))
end

mutable struct vb_state
    iter::Int
    F::Float64
    λ::Vector{Float64}
    ϵ_θ::Vector{Float64}
    μθ_po::Vector{Float64}
    Σθ_po::Matrix{Float64}
end

function vecparam(param::OrderedDict{Any,Any})
    flatparam = Float64[]
    for v in values(param)
        if (typeof(v) <: Array)
            for vv in v
                push!(flatparam, vv)
            end
        else
            push!(flatparam, v)
        end
    end
    return flatparam
end

function unvecparam(vals, param::OrderedDict{Any,Any})
    iter = 1
    paramnewvals = copy(param)
    for (k, v) in param
        if (typeof(v) <: Array)
            paramnewvals[k] = vals[iter:iter+length(v)-1]
            iter += length(v)
        else
            paramnewvals[k] = vals[iter]
            iter += 1
        end
    end
    return paramnewvals
end


# """
#     function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, indices)

#     Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
#     The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

#     Arguments:
#     - `data`        : dataframe with column names corresponding to the regions of measurement.
#     - `model`       : MTK model, including state evolution and measurement.
#     - `initcond`    : dictionary of initial conditions, numerical values for all states
#     - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
#     -- `dt`         : sampling interval
#     -- `freq`       : frequencies at which to evaluate the CSD
#     -- `p`          : order parameter of the multivariate autoregression model
#     - `priors`      : dataframe of parameters with the following columns:
#     -- `name`       : corresponds to MTK model name
#     -- `mean`       : corresponds to prior mean value
#     -- `variance`   : corresponds to the prior variances
#     - `hyperpriors` : dataframe of parameters with the following columns:
#     -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
#     -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
#     - `indices`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
# """
# function setup_sDCM(data, model, initcond, csdsetup, priors, hyperpriors, indices, modality)
#     # compute cross-spectral density
#     dt = csdsetup.dt;              # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
#     freq = csdsetup.freq;                # frequencies at which the CSD is evaluated
#     mar_order = csdsetup.mar_order;        # order of MAR
#     _, vars = get_eqidx_tagged_vars(model, "measurement")
#     data = Matrix(data[:, Symbol.(vars)])  # make sure the column order is consistent with the ordering of variables of the model that represent the measurements
#     mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
#     y_csd = mar2csd(mar, freq, dt^-1);        # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
#     jac_fg = generate_jacobian(model, expression = Val{false})[1]   # compute symbolic jacobian.

#     statevals = [v for v in values(initcond)]
#     derivatives = par -> jac_fg(statevals, addnontunableparams(par, model), t)

#     μθ_pr = vecparam(OrderedDict(priors.name .=> priors.mean))            # note: μθ_po is posterior and μθ_pr is prior
#     Σθ_pr = diagm(vecparam(OrderedDict(priors.name .=> priors.variance)))

#     ### Collect prior means and covariances ###
#     if haskey(hyperpriors, :Q)
#         Q = hyperpriors[:Q];
#     else
#         Q = csd_Q(y_csd);                 # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
#     end
#     nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
#     nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)

#     f = params -> csd_mtf(freq, mar_order, derivatives, params, indices, modality)

#     np = length(μθ_pr)     # number of parameters
#     ny = length(y_csd)     # total number of response variables

#     # variational laplace state variables
#     vlstate = VLState(
#         0,                                   # iter
#         -4,                                  # log ascent rate
#         [-Inf],                              # free energy
#         Float64[],                           # delta free energy
#         hyperpriors[:μλ_pr],                 # metaparameter, initial condition. TODO: why are we not just using the prior mean?
#         zeros(np),                           # parameter estimation error ϵ_θ
#         [zeros(np), hyperpriors[:μλ_pr]],    # memorize reset state
#         μθ_pr,                               # parameter posterior mean
#         Σθ_pr,                               # parameter posterior covariance
#         zeros(np),
#         zeros(np, np)
#     )

#     # variational laplace setup
#     vlsetup = VLSetup(
#         f,                                    # function that computes the cross-spectral density at fixed point 'initcond'
#         y_csd,                                # empirical cross-spectral density
#         1e-1,                                 # tolerance
#         [np, ny, nq, nh],                     # number of parameters, number of data points, number of Qs, number of hyperparameters
#         [μθ_pr, hyperpriors[:μλ_pr]],         # parameter and hyperparameter prior mean
#         [inv(Σθ_pr), hyperpriors[:Πλ_pr]],    # parameter and hyperparameter prior precision matrices
#         Q                                     # components of data precision matrix
#     )
#     return (vlstate, vlsetup)
# end

# function run_sDCM_iteration!(state::VLState, setup::VLSetup)
#     (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

#     f = setup.model_at_x0
#     y = setup.y_csd              # cross-spectral density
#     (np, ny, nq, nh) = setup.systemnums
#     (μθ_pr, μλ_pr) = setup.systemvecs
#     (Πθ_pr, Πλ_pr) = setup.systemmatrices
#     Q = setup.Q

#     dFdθ = jacobian(f, μθ_po)

#     norm_dFdθ = opnorm(dFdθ, Inf);
#     revert = isnan(norm_dFdθ) || norm_dFdθ > exp(32);

#     if revert && state.iter > 1
#         for i = 1:4
#             # reset expansion point and increase regularization
#             v = min(v - 2, -4);
#             t = exp(v - logdet(dFdθθ)/np)

#             # E-Step: update
#             if t > exp(16)
#                 ϵ_θ = ϵ_θ - dFdθθ \ dFdθ    # -inv(dfdx)*f
#             else
#                 ϵ_θ = ϵ_θ + expv(t, dFdθθ, dFdθθ \ dFdθ) - dFdθθ \ dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
#             end

#             μθ_po = μθ_pr + ϵ_θ

#             dFdθ = jacobian(f, μθ_po)

#             # check for stability
#             norm_dFdθ = opnorm(dFdθ, Inf);
#             revert = isnan(norm_dFdθ) || norm_dFdθ > exp(32);

#             # break
#             if ~revert
#                 break
#             end
#         end
#     end

#     ϵ = reshape(y - f(μθ_po), ny)                   # error
#     J = - dFdθ   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

#     ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
#     P = zeros(eltype(J), size(Q))
#     PΣ = zeros(eltype(J), size(Q))
#     JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
#     dFdλ = zeros(eltype(J), nh)
#     dFdλλ = zeros(real(eltype(J)), nh, nh)
#     local iΣ, Σλ_po, Σθ_po, ϵ_λ
#     for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
#         iΣ = zeros(eltype(J), ny, ny)
#         for i = 1:nh
#             iΣ .+= Q[:, :, i] * exp(λ[i])
#         end

#         Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
#         Σθ_po = inv(Pp + Πθ_pr)

#         for i = 1:nh
#             P[:,:,i] = Q[:,:,i]*exp(λ[i])
#             PΣ[:,:,i] = iΣ \ P[:,:,i]
#             JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
#         end
#         for i = 1:nh
#             dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
#             for j = i:nh
#                 dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
#                 dFdλλ[j, i] = dFdλλ[i, j]
#             end
#         end

#         ϵ_λ = λ - μλ_pr
#         dFdλ = dFdλ - Πλ_pr*ϵ_λ
#         dFdλλ = dFdλλ - Πλ_pr
#         Σλ_po = inv(-dFdλλ)

#         t = exp(4 - spm_logdet(dFdλλ)/length(λ))
#         # E-Step: update
#         if t > exp(16)
#             dλ = -real(dFdλλ \ dFdλ)
#         else
#             idFdλλ = inv(dFdλλ)
#             dλ = real(exponential!(t * dFdλλ) * idFdλλ*dFdλ - idFdλλ*dFdλ)   # (expm(dfdx*t) - I)*inv(dfdx)*f ~~~ could also be done with expv but doesn't work with Dual.
#         end

#         dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
#         λ = λ + dλ

#         dF = dot(dFdλ, dλ)

#         # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
#         # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
#         # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
#         if real(dF) < 1e-2
#             break
#         end
#     end

#     ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
#     L = zeros(real(eltype(iΣ)), 3)
#     L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
#     L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
#     L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
#     F = sum(L)

#     if F > state.F[end] || state.iter < 3
#         # accept current state
#         state.reset_state = [ϵ_θ, λ]
#         append!(state.F, F)
#         state.Σθ_po = Σθ_po
#         # Conditional update of gradients and curvature
#         dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
#         dFdθθ = -real(J' * iΣ * J) - Πθ_pr
#         # decrease regularization
#         v = min(v + 1/2, 4);
#     else
#         # reset expansion point
#         ϵ_θ, λ = state.reset_state
#         # and increase regularization
#         v = min(v - 2, -4);
#     end

#     # E-Step: update
#     t = exp(v - spm_logdet(dFdθθ)/np)
#     if t > exp(16)
#         dθ = - inv(dFdθθ) * dFdθ     # -inv(dfdx)*f
#     else
#         dθ = exponential!(t * dFdθθ) * inv(dFdθθ) * dFdθ - inv(dFdθθ) * dFdθ     # (expm(dfdx*t) - I)*inv(dfdx)*f
#     end

#     ϵ_θ += dθ
#     state.μθ_po = μθ_pr + ϵ_θ
#     dF = dot(dFdθ, dθ);

#     state.v = v
#     state.ϵ_θ = ϵ_θ
#     state.λ = λ
#     state.dFdθθ = dFdθθ
#     state.dFdθ = dFdθ
#     append!(state.dF, dF)

#     return state
# end

function variationalbayes(x, y, w, V, p, priors, niter)    # relates to spm_nlsi_GN.m
    # extract priors
    Πθ_pr = priors[:Σ][:Πθ_pr]
    Πλ_pr = priors[:Σ][:Πλ_pr]
    μλ_pr = priors[:Σ][:μλ_pr]
    Q = priors[:Σ][:Q]

    # prep stuff
    μθ_pr = vecparam(priors[:μ])            # note: μθ_po is posterior and μθ_pr is prior
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    # ns = size(y, 1)            # number of samples
    # nr = ny÷ns                 # number of response components
    nq = 1
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)
    ϵ_θ = zeros(np)  # M.P - μθ_pr # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ_po with μθ_pr above.
    μθ_po = μθ_pr + V*ϵ_θ

    dx = exp(-8)
    revert = false
    f_prep = param -> csd_fmri_mtf(x, w, p, param)

    # state variable
    F = -Inf
    F0 = F
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    state = vb_state(0, F, λ, zeros(np), μθ_po, inv(Πθ_pr))
    local ϵ_λ, iΣ, Σλ_po, Σθ_po, dFdθθ, dFdθ
    dFdλ = zeros(Float64, nh)
    dFdλλ = zeros(Float64, nh, nh)
    for k = 1:niter
        state.iter = k

        dFdθ, f = diff(V, dx, f_prep, μθ_po);
        dFdθ = transpose(reshape(dFdθ, np, ny))
        norm_dFdθ = opnorm(dFdθ, Inf);
        revert = isnan(norm_dFdθ) || norm_dFdθ > exp(32);

        if revert && k > 1
            for i = 1:4
                # reset expansion point and increase regularization
                v = min(v - 2, -4);
                t = exp(v - logdet(dFdθθ)/np)

                # E-Step: update
                if t > exp(16)
                    ϵ_θ = state.ϵ_θ - inv(dFdθθ)*dFdθ    # -inv(dfdx)*f
                else
                    ϵ_θ = state.ϵ_θ + expv(t, dFdθθ, inv(dFdθθ)*dFdθ) -inv(dFdθθ)*dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
                end

                μθ_po = μθ_pr + V*ϵ_θ

                dFdθ, f = diff(V, dx, f_prep, μθ_po);
                dFdθ = transpose(reshape(dFdθ, np, ny))

                # check for stability
                norm_dFdθ = opnorm(dFdθ, Inf);
                revert = isnan(norm_dFdθ) || norm_dFdθ > exp(32);

                # break
                if ~revert
                    break
                end
            end
        end


        ϵ = reshape(y - f, ny)                   # error value
        J = - dFdθ   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 


        ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
        P = similar(Q)
        PΣ = similar(Q)
        JPJ = zeros(size(J, 2), size(J, 2), size(Q, 3))
        for m = 1:8   # 8 seems arbitrary. This is probably because optimization falls often into a periodic orbit. ToDo: Issue #8
            iΣ = zeros(ComplexF64, ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end

            Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
            Σθ_po = inv(Pp + Πθ_pr)
            if nh > 1
                for i = 1:nh
                    P[:,:,i] = Q[:,:,i]*exp(λ[i])
                    PΣ[:,:,i] = real(iΣ \ P[:,:,i])
                    JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
                end
                for i = 1:nh
                    dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
                    for j = i:nh
                        dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                        dFdλλ[j, i] = dFdλλ[i, j]
                    end
                end
            else
                dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;
                dFdλλ[1, 1] = - ny/2;
            end

            dFdλλ = dFdλλ + diagm(dFdλ);

            ϵ_λ = λ - μλ_pr
            dFdλ = dFdλ - Πλ_pr*ϵ_λ
            dFdλλ = dFdλλ - Πλ_pr
            Σλ_po = inv(-dFdλλ)

            t = exp(4 - spm_logdet(dFdλλ)/length(λ))
            # E-Step: update
            if t > exp(16)
                dλ = -real(inv(dFdλλ) * dFdλ)
            else
                dλ = real(expv(t, dFdλλ, inv(dFdλλ)*dFdλ) -inv(dFdλλ)*dFdλ)   # (expm(dfdx*t) - I)*inv(dfdx)*f
            end

            dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
            λ = λ + dλ

            dF = dot(dFdλ, dλ)

            if real(dF) < 1e-2
                break
            end
        end

        ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
        L = zeros(3)
        L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
        L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
        L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
        F = sum(L);

        if k == 1
            F0 = F
        end

        if F > state.F || k < 3
            # accept current state
            state.F = F
            state.ϵ_θ = ϵ_θ
            state.λ = λ
            state.Σθ_po = Σθ_po
            state.μθ_po = μθ_po
            # Conditional update of gradients and curvature
            dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ
            dFdθθ = -real(J' * iΣ * J) - Πθ_pr
            # decrease regularization
            v = min(v + 1/2,4);
        else
            # reset expansion point
            ϵ_θ = state.ϵ_θ
            λ = state.λ
            # and increase regularization
            v = min(v - 2,-4);
        end

        # E-Step: update
        t = exp(v - spm_logdet(dFdθθ)/np)
        if t > exp(16)
            dθ = - inv(dFdθθ)*dFdθ    # -inv(dfdx)*f
        else
            dθ = exp(t * dFdθθ) * inv(dFdθθ)*dFdθ - inv(dFdθθ)*dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
        end

        ϵ_θ += dθ
        μθ_po = μθ_pr + V*ϵ_θ
        dF  = dot(dFdθ, dθ);

        # convergence condition: reach a change in Free Energy that is smaller than 0.1 four consecutive times
        print("iteration: ", k, " - F:", state.F - F0, " - dF predicted:", dF, "\n")
        criterion = vcat(dF < 1e-1, criterion[1:end - 1]);
        if all(criterion)
            print("convergence\n")
            break
        end
    end
    print("iterations terminated\n")
    state.F = F
    state.Σθ_po = V*Σθ_po*V'
    state.μθ_po = μθ_po

    return state
end