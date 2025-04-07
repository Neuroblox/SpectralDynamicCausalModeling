using ForwardDiff: Dual, Partials, jacobian
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true

tagtype(::Dual{T,V,N}) where {T,V,N} = T

include("utils/helperfunctions.jl")
include("utils/helperfunctions_AD.jl")

"""
VariationalBayes_MTKAD.jl

Main functions to compute a spectral DCM.

transferfunction : computes transfer function of neuronal model as well as measurement model
csd_approx       : approximates CSD based on transfer functions
csd_fmri_mtf     :
csd_Q            : computes precision component prior (which erroneously is not used in the SPM12 code for fMRI signals, it is used for other modalities)
spm_logdet       : mimick SPM's way to compute the logarithm of the determinant. Sometimes Julia's logdet won't work.
variationalbayes : main routine that computes the variational Bayes estimate of model parameters
"""


function transferfunction(ω, derivatives, params, indices)
    ∂f = derivatives(params[indices[:dspars]])
    # dissect into Jacobian w.r.t. dynamic variables as well as partial derivatives wrt input variables and gradient of measurement function
    ∂f∂x = ∂f[indices[:sts], indices[:sts]]
    ∂f∂u = ∂f[indices[:sts], indices[:u]]
    ∂g∂x = ∂f[indices[:m], indices[:sts]]

    F = eigen(∂f∂x)
    Λ = F.values
    V = F.vectors

    ∂g∂v = ∂g∂x*V
    ∂v∂u = V\∂f∂u              # u is external variable.

    nω = size(ω, 1)            # number of frequencies
    ng = size(∂g∂x, 1)         # number of outputs
    nu = size(∂v∂u, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(∂v∂u))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*ω .- Λ[k]).^-1
                S[:,i,j] .+= ∂g∂v[i,k]*∂v∂u[k,j]*Sk
            end
        end
    end

    return S
end


"""
    This function implements equation 2 of the spectral DCM paper, Friston et al. 2014 "A DCM for resting state fMRI".
    Note that nomenclature is taken from SPM12 code and it does not seem to coincide with the spectral DCM paper's nomenclature. 
    For instance, Gu should represent the spectral component due to external input according to the paper. However, in the code this represents
    the hidden state fluctuations (which are called Gν in the paper).
    Gn in the code corresponds to Ge in the paper, i.e. the observation noise. In the code global and local components are defined, no such distinction
    is discussed in the paper. In fact the parameter γ, corresponding to local component is not present in the paper.
"""
function csd_approx(ω, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nω = length(ω)
    nd = length(params_idx[:lnγ])
    α = params[params_idx[:lnα]]
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]
    
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    G = ω.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nω, nd, nd)
    Gn = zeros(eltype(G), nω, nd, nd)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1])*G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = ω.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nd
        Gn[:,i,i] .+= exp(γ[i])*G
    end

    # global components
    for i = 1:nd
        for j = i:nd
            Gn[:,i,j] .+= exp(β[1])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    S = transferfunction(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end


function csd_approx_lfp(ω, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nω = length(ω)
    nd = length(params_idx[:lnγ])
    α = reshape(params[params_idx[:lnα]], nd, nd)
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]

    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".
    Gu = zeros(eltype(α), nω, nd)   # spectrum of neuronal innovations or intrinsic noise or system noise
    Gn = zeros(eltype(β), nω)   # global spectrum of channel noise or observation noise or external noise
    Gs = zeros(eltype(γ), nω)   # region specific spectrum of channel noise or observation noise or external noise
    for i = 1:nd
        Gu[:, i] .+= exp(α[1, i]) .* ω.^(-exp(α[2, i]))
    end
    # global components and region specific observation noise (1/f or AR(1) form)
    Gn = exp(β[1] - 2) * ω.^(-exp(β[2]))
    Gs = exp(γ[1] - 2) * ω.^(-exp(γ[2]))  # this is really oddly implemented in SPM12. Completely unclear how this should be region specific

    S = transferfunction(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*diagm(Gu[i,:])*S[i,:,:]'
    end

    for i = 1:nd
        G[:,i,i] += Gs
        for j = 1:nd
            G[:,i,j] += Gn
        end
    end

    return G
end

@views function csd_mtf(freqs, p, derivatives, params, params_idx, modality)   # alongside the above realtes to spm_csd_fmri_mtf.m
    if modality == "fMRI"
        G = csd_approx(freqs, derivatives, params, params_idx)

        dt = 1/(2*freqs[end])
        # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
        # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
        # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. NB: Friston conferms that likely it is
        # to make y well behaved.
        mar = csd2mar(G, freqs, dt, p-1)
        y = mar2csd(mar, freqs)    
    elseif modality == "LFP"
        y = csd_approx_lfp(freqs, derivatives, params, params_idx)
    end
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end

    return y
end

"""
    function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, indices)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

    Arguments:
    - `data`        : dataframe with column names corresponding to the regions of measurement.
    - `model`       : MTK model, including state evolution and measurement.
    - `initcond`    : ordered dictionary of initial conditions: make sure that the ordering corresponds to the order of the model unknowns/variables
    - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
    -- `dt`         : sampling interval
    -- `freq`       : frequencies at which to evaluate the CSD
    -- `p`          : order parameter of the multivariate autoregression model
    - `priors`      : dataframe of parameters with the following columns:
    -- `name`       : corresponds to MTK model name
    -- `mean`       : corresponds to prior mean value
    -- `variance`   : corresponds to the prior variances
    - `hyperpriors` : dataframe of parameters with the following columns:
    -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
    - `indices`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
"""
function setup_spDCM(data, model, initcond, csdsetup, priors, hyperpriors, indices, modelparam, modality)
    # compute cross-spectral density
    dt = csdsetup.dt;                      # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freq = csdsetup.freq;                  # frequencies at which the CSD is evaluated
    mar_order = csdsetup.mar_order;        # order of MAR
    mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freq, dt^-1);     # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    statevals = [v for v in values(initcond)]
    append!(statevals, zeros(length(unknowns(model)) - length(statevals)))
    f_model = generate_function(model; expression=Val{false})[1]
    f_at(params, t) = states -> f_model(states, MTKParameters(model, params)..., t)
    derivatives = par -> jacobian(f_at(addnontunableparams(par, model), t), statevals)

    μθ_pr = vecparam(priors.μθ_pr)        # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = diagm(vecparam(priors.Σθ_pr))

    ### Collect prior means and covariances ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = csd_Q(y_csd);             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)
    nr = length(indices[:lnγ])        # region specific noise parameter can be used to get the number of regions

    f = params -> csd_mtf(freq, mar_order, derivatives, params, indices, modality)

    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLState(
        0,                                   # iter
        -4,                                  # log ascent rate
        [-Inf],                              # free energy
        Float64[],                           # delta free energy
        hyperpriors.μλ_pr,                   # metaparameter, initial condition.
        zeros(np),                           # parameter estimation error ϵ_θ
        [zeros(np), hyperpriors.μλ_pr],      # memorize reset state
        μθ_pr,                               # parameter posterior mean
        Σθ_pr,                               # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLMTKSetup(
        f,                                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                                # empirical cross-spectral density
        1e-1,                                 # tolerance
        [nr, np, ny, nq, nh],                 # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, hyperpriors.μλ_pr],           # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors.Πλ_pr],      # parameter and hyperparameter prior precision matrices
        Q,                                    # components of data precision matrix
        modelparam
    )
    return (vlstate, vlsetup)
end


"""
    variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)

    Computes parameter estimation using variational Laplace that is to a large extend equivalent to the SPM12 implementation
    and provides the exact same values.

    Arguments:
    - `idx_A`: indices of connection weight parameter matrix A in model Jacobian
    - `y`: empirical cross-spectral density (input data)
    - `derivatives`: jacobian of model as well as gradient of observer function
    - `w`: fequencies at which to estimate cross-spectral densities
    - `V`: projection matrix from full parameter space to reduced space that removes parameters with zero variance prior
    - `p`: order of multivariate autoregressive model for estimation of cross-spectral densities from data
    - `priors`: Bayesian priors, mean and variance thereof. Laplace approximation assumes Gaussian distributions
    - `niter`: number of iterations of the optimization procedure
"""
function run_spDCM_iteration!(state::VLState, setup::VLMTKSetup)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (nr, np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)

    norm_dfdp = opnorm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);

            # E-Step: update
            ϵ_θ += integration_step(dFdθθ, dFdθ, v)

            μθ_po = μθ_pr + ϵ_θ

            dfdp = ForwardDiff.jacobian(f, μθ_po)

            # check for stability
            norm_dfdp = opnorm(dfdp, Inf);
            revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f(μθ_po), ny)                   # error
    J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(real(eltype(J)), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        if nh > 1
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = iΣ \ P[:,:,i]
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
            # if nh == 1, do the followng simplifications to improve computational speed:          
            # 1. replace trace(PΣ[1]) * nq by ny
            # 2. replace JPJ[1] by Pp
            dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;

            # 3. replace trace(PΣ[1],PΣ[1]) * nq by ny
            dFdλλ[1, 1] = - ny/2;
        end

        dFdλλ = dFdλλ + diagm(dFdλ);      # add second order terms; noting diΣ/dλ(i)dλ(i) = diΣ/dλ(i) = P{i}

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        # E-Step: update
        dλ = real(integration_step(dFdλλ, dFdλ, 4))

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    dθ = integration_step(dFdθθ, dFdθ, v, true)

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end