using LinearAlgebra: Eigen
using ChainRules: _eigen_norm_phase_fwd!
using ForwardDiff: Dual, Partials, jacobian
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true


"""
    function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}

    Dispatch of LinearAlgebra.eigen for dual matrices with complex numbers. Make the eigenvalue decomposition 
    amenable to automatic differentiation. To do so compute the analytical derivative of eigenvalues
    and eigenvectors. 

    Arguments:
    - `M`: matrix of type Dual of which to compute the eigenvalue decomposition. 

    Returns:
    - `Eigen(evals, evecs)`: eigenvalue decomposition returned as type LinearAlgebra.Eigen
"""
function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    nd = size(M, 1)
    A = (p->p.value).(M)
    F = eigen(A, sortby=nothing, permute=true)
    λ, V = F
    local ∂λ_agg, ∂V_agg
    # compute eigenvalue and eigenvector derivatives for all partials
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)   # copy only needed for Complex because `real.(v)` makes a new array
        ∂K ./= transpose(λ) .- λ
        fill!(∂Kdiag, 0)
        ∂V_tmp = mul!(tmp, V, ∂K)
        _eigen_norm_phase_fwd!(∂V_tmp, A, V)
        if i == 1
            ∂V_agg = ∂V_tmp
            ∂λ_agg = ∂λ_tmp
        else
            ∂V_agg = cat(∂V_agg, ∂V_tmp, dims=3)
            ∂λ_agg = cat(∂λ_agg, ∂λ_tmp, dims=2)
        end
    end
    # reassemble the aggregated vectors and values into a Partials type
    ∂V = map(Iterators.product(1:nd, 1:nd)) do (i, j)
        Partials(NTuple{np}(∂V_agg[i, j, :]))
    end
    ∂λ = map(1:nd) do i
        Partials(NTuple{np}(∂λ_agg[i, :]))
    end
    if eltype(V) <: Complex
        evals = map(λ, ∂λ) do x, y
            rex, imx = reim(x)
            rey, imy = real.(Tuple(y)), imag.(Tuple(y))
            Complex(Dual{T}(rex, Partials(rey)), Dual{T}(imx, Partials(imy)))
        end
        evecs = map(V, ∂V) do x, y
            rex, imx = reim(x)
            rey, imy = real.(Tuple(y)), imag.(Tuple(y))
            Complex(Dual{T}(rex, Partials(rey)), Dual{T}(imx, Partials(imy)))
        end
    else
        evals = Dual{T}.(λ, ∂λ)
        evecs = Dual{T}.(V, ∂V)
    end
    return Eigen(evals, evecs)
end