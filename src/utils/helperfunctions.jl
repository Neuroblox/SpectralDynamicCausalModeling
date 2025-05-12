"""
    integration_step(dfdx, f, v, solenoid=false)

"""
function integration_step(J, f, v, solenoid=false)
    if solenoid
        # add solenoidal mixing as is present in the later versions of SPM, in particular SPM25
        L  = tril(J);
        Q  = L - L';
        Q  = Q/opnorm(Q, 2)/8;

        f  = f - Q*f;
        J = J - Q*J;
    end

    # NB: (exp(dfdx*t) - I)*inv(dfdx)*f, could also be done with expv (expv(t, dFdθθ, dFdθθ \ dFdθ) - dFdθθ \ dFdθ) but doesn't work with Dual.
    # Could also be done with `exponential!` but isn't numerically stable.
    # Thus, just use `exp`.
    n = length(f)
    t = exp(v - spm_logdet(real(J))/n)

    if t > exp(16)
        dx = - J \ f
    else
        dx = (exp(t * J) - I) * inv(J) * f
    end

    return dx
end

"""
    function removezerovardims(M)
    Note that this helper function is used to remove dimensions with zero variance from the prior covariance matrix since they are fixed.
    In SPM covariance matrices with non-diagonal elements will undergo an SVD decomposition, here we only deal with diagonal matrices for now.
"""
function removezerovardims(M)
    idx = findall(x -> x != 0, M);
    V = zeros(size(M, 1), length(idx));
    order = sortperm(M[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    A = V'*M*V;       # redimension matrix by removing columns and rows that have zero entries

    return (A, V)
end
"""
    vecparam(param::OrderedDict)

    Function to flatten an ordered dictionary of model parameters and return a simple list of parameter values.

    Arguments:
    - `param`: dictionary of model parameters (may contain numbers and lists of numbers)
"""
function vecparam(param::OrderedDict)
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

"""
    function spm_logdet(M)

    SPM12 style implementation of the logarithm of the determinant of a matrix.

    Arguments:
    - `M`: matrix
"""
function spm_logdet(M)
    TOL = 1e-16
    s = diag(M)
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
end

"""
    function csd_Q(csd)

    Compute correlation matrix to be used as functional connectivity prior.
"""
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
    Q = inv(Q .+ opnorm(Q, 1)/32*Matrix(I, size(Q)))
    return Q
end
