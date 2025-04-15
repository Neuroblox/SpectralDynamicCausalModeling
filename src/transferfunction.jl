using ForwardDiff: Dual
tagtype(::Dual{T,V,N}) where {T,V,N} = T


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
                Sk = (1im*2*pi*ω .- Λ[k]).^-1
                S[:,i,j] .+= dgdv[i,k]*dvdu[k,j]*Sk
            end
        end
    end

    return S
end

function csd_fmri_mtf(x, ω, p, params)
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
            Gn[:, j, i] = Gn[:, i, j]
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
    y = mar2csd(mar, ω)
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    return y
end
