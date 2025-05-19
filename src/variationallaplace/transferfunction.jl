tagtype(::Dual{T,V,N}) where {T,V,N} = T


function _transferfunction(ω, F, ∂f∂x, ∂f∂u, ∂g∂x)
    Λ = F.values
    V = F.vectors

    ∂g∂v = ∂g∂x*V
    ∂v∂u = V\∂f∂u              # u is external variable.

    nω = size(ω, 1)            # number of frequencies
    ng = size(∂g∂x, 1)         # number of outputs
    nu = size(∂v∂u, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(∂g∂x))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = ((2im*pi) .* ω .- Λ[k]).^-1
                S[:,i,j] .+= ∂g∂v[i,k]*∂v∂u[k,j]*Sk
            end
        end
    end
    return S
end

function transferfunction(ω, derivatives, params, indices)
    ∂f = derivatives(params[indices[:dspars]])
    # dissect into Jacobian w.r.t. dynamic variables as well as partial derivatives wrt input variables and gradient of measurement function
    ∂f∂x = ∂f[indices[:sts], indices[:sts]]
    ∂f∂u = ∂f[indices[:sts], indices[:u]]
    ∂g∂x = ∂f[indices[:m], indices[:sts]]
    F = eigen(∂f∂x)
    S = _transferfunction(ω, F, ∂f∂x, ∂f∂u, ∂g∂x)
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
    nd = length(params_idx[:u])
    α = reshape(params[params_idx[:lnα]], 2, nd)
    β = params[params_idx[:lnβ]]
    γ = reshape(params[params_idx[:lnγ]], 2, nd)

    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".
    Gu = zeros(eltype(α), nω, nd)   # spectrum of neuronal innovations or intrinsic noise or system noise
    Gn = zeros(eltype(β), nω)       # global spectrum of channel noise or observation noise or external noise
    Gs = zeros(eltype(γ), nω, nd)   # region specific spectrum of channel noise or observation noise or external noise
    for i = 1:nd
        Gu[:, i] .+= exp(α[1, i]) .* ω.^(-exp(α[2, i]))
    end
    # global components and region specific observation noise (1/f or AR(1) form)
    Gn = exp(β[1] - 2) * ω.^(-exp(β[2]))
    for i = 1:nd
        Gs[:, i] .+= exp(γ[1, i] - 2) .* ω.^(-exp(γ[2, 1]))  # this is really oddly implemented in SPM, the exponent parameter is kept fixed, leaving parameters that essentially don't matter
    end

    S = transferfunction(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*diagm(Gu[i,:])*S[i,:,:]'
    end

    for i = 1:nd
        G[:,i,i] += Gs[:,i]
        for j = 1:nd
            G[:,i,j] += Gn
        end
    end

    return G
end

function csd_mtf!(y, freqs, p, derivatives, params, params_idx, modality)   # alongside the above realtes to spm_csd_fmri_mtf.m
    if modality == "fMRI"
        G = csd_approx(freqs, derivatives, params, params_idx)

        dt = 1/(2*freqs[end])
        # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
        # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
        # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. NB: Friston conferms that likely it is
        # to make y well behaved.
        mar = csd2mar(G, freqs, dt, p-1)
        csd = mar2csd(mar, freqs)
    elseif modality == "LFP"
        csd = csd_approx_lfp(freqs, derivatives, params, params_idx)
    end
    # convert complex of duals to dual of complex:
    if real(eltype(csd)) <: Dual
        y[:,:,:] = map(csd) do csdi
            csdi_val  = Complex(real(csdi).value, imag(csdi).value)
            csdi_part = Partials(Complex.(real(csdi).partials.values, imag(csdi).partials.values))
            Dual{tagtype(real(csdi))}(csdi_val, csdi_part)
        end
    else
        y[:,:,:] = csd
    end

    return y
end

function transferfunction(x, ω, params)
    # compute transfer function of Volterra kernels, see fig 1 in friston2014
    C = diagm(params.C/16.0)   # division by 16 taken from SPM, see spm_fx_fmri.m:49
    J = hemodynamics_jacobian(x[:, 2:end], params.lnκ, params.lnτ)
    θμ = params.A
    θμ -= diagm(exp.(diag(θμ))/2 + diag(θμ))
    nd = size(θμ, 1)
    J_tot = [θμ zeros(nd, size(J, 2));   # add derivatives w.r.t. neural signal
             [Matrix(1.0I, size(θμ)); zeros(size(J)[1]-nd, size(θμ)[2])] J]

    dfdu = [C;
            zeros(size(J,1), size(C, 2))]

    F = eigen(J_tot)
    Λ = F.values
    V = F.vectors

    dgdx = boldsignal(x, params.lnϵ)[2]
    dgdv = dgdx * @view V[end-size(dgdx,2)+1:end, :]
    dvdu = V\dfdu

    nω = size(ω, 1)            # number of frequencies
    ng = size(dgdx, 1)         # number of outputs
    nu = size(dfdu, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(dvdu))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = ((2im*pi) .*ω .- Λ[k]).^-1
                S[:, i, j] .+= dgdv[i, k]*dvdu[k, j]*Sk
            end
        end
    end

    return S
end

function csd_fmri_mtf!(y, x, ω, p, params)
    nω = length(ω)
    nr = size(params.A, 1)

    # define function that implements spectra given in equation (2) of "A DCM for resting state fMRI".
    # neuronal fluctuations (Gu) (1/f or AR(1) form)
    G = ω.^(-exp(params.lnα[2]))   # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nω, nr, nr)
    Gn = zeros(eltype(G), nω, nr, nr)
    for i = 1:nr
        Gu[:, i, i] .+= exp(params.lnα[1])*G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = ω.^(-exp(params.lnβ[2])/2)
    G /= sum(G)
    for i = 1:nr
        Gn[:, i, i] .+= exp(params.lnγ[i])*G
    end

    # global components
    for i = 1:nr
        for j = i:nr
            Gn[:, i, j] .+= exp(params.lnβ[1])*G
            Gn[:, j, i] .= Gn[:, i, j]
        end
    end
    S = transferfunction(x, ω, params)
    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nr, nr);
    for i = 1:nω
        G[i, :, :] = S[i, :, :]*Gu[i, :, :]*S[i, :, :]'
    end
    G_final = G + Gn

    dt  = 1/(2*ω[end])
    # this transformation and back-transformation is taken from SPM. This is likely to stabilize the results, mathematically there is no reason 
    # for doing this and this operation is not present in the spDCM papers.
    mar = csd2mar(G_final, ω, dt, p-1)
    csd = mar2csd(mar, ω)
    # convert complex of duals to dual of complex:
    if real(eltype(csd)) <: Dual
        y[:,:,:] = map(csd) do csdi
            csdi_val  = Complex(real(csdi).value, imag(csdi).value)
            csdi_part = Partials(Complex.(real(csdi).partials.values, imag(csdi).partials.values))
            Dual{tagtype(real(csdi))}(csdi_val, csdi_part)
        end
    else
        y[:,:,:] = csd
    end
    return y
end
