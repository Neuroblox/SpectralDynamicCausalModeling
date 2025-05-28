
Base.copy(S::VLMTKState) = VLMTKState(
    copy(S.iter),
    copy(S.v),
    copy(S.F),
    copy(S.dF),
    copy(S.λ),
    copy(S.ϵ_θ),
    copy(S.reset_state),
    copy(S.μθ_po),
    copy(S.Σθ_po),
    copy(S.dFdθ),
    copy(S.dFdθθ)
)


function defaultprior(model, nr)
    _, idx_sts = get_dynamic_states(model)
    idx_u = get_idx_tagged_vars(model, "ext_input")                  # get index of external input state
    idx_bold, _ = get_eqidx_tagged_vars(model, "measurement")        # get index of equation of bold state

    # collect parameter default values, these constitute the prior mean.
    # parcomp = "ComponentArray("
    # parcomp *= string((par -> String(par) * " = " * string(Symbolics.getdefaultval(par)) * ",").(Symbol.(tunable_parameters(model)))...)
    # parcomp *= ")"

    parammean = OrderedDict()
    np = sum(tunable_parameters(model); init=0) do par
        val = Symbolics.getdefaultval(par)
        parammean[par] = val
        length(val)
    end
    indices = Dict(:dspars => collect(1:np))

    # Noise parameters
    parammean[:lnα] = [0.0, 0.0];            # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
    n = length(parammean[:lnα]);
    indices[:lnα] = collect(np+1:np+n);
    np += n;
    parammean[:lnβ] = [0.0, 0.0];            # global observation noise, ln(β) as above
    n = length(parammean[:lnβ]);
    indices[:lnβ] = collect(np+1:np+n);
    np += n;
    parammean[:lnγ] = zeros(Float64, nr);    # region specific observation noise
    indices[:lnγ] = collect(np+1:np+nr);
    np += nr
    indices[:u] = idx_u
    indices[:m] = idx_bold
    indices[:sts] = idx_sts

    # continue with prior variances
    paramvariance = copy(parammean)
    paramvariance[:lnγ] = ones(Float64, nr)./64.0;
    paramvariance[:lnα] = ones(Float64, length(parammean[:lnα]))./64.0;
    paramvariance[:lnβ] = ones(Float64, length(parammean[:lnβ]))./64.0;
    for (k, v) in paramvariance
        if occursin("A", string(k))
            paramvariance[k] = ones(length(v))./64.0;
        elseif occursin("κ", string(k))
            paramvariance[k] = ones(length(v))./256.0;
        elseif occursin("ϵ", string(k))
            paramvariance[k] = 1/256.0;
        elseif occursin("τ", string(k))
            paramvariance[k] = 1/256.0;
        end
    end
    return parammean, diagm(vecparam(paramvariance)), indices
end

"""
    function get_dynamic_states(sys)
    
    Function extracts states from the system that are dynamic variables, 
    get also indices of external inputs (u(t)) and measurements (like bold(t))
    Arguments:
    - `sys`: MTK system

    Returns:
    - `sts`: states/unknowns of the system that are neither external inputs nor measurements, i.e. these are the dynamic states
    - `idx`: indices of these states
"""
function get_dynamic_states(sys)
    itr = Iterators.filter(enumerate(unknowns(sys))) do (_, s)
        !((getdescription(s) == "ext_input") || (getdescription(s) == "measurement"))
    end
    sts = map(x -> x[2], itr)
    idx = map(x -> x[1], itr)
    return sts, idx
end

function get_eqidx_tagged_vars(sys, tag)
    idx = Int[]
    vars = []
    eqs = equations(sys)
    for s in unknowns(sys)
        if getdescription(s) == tag
            push!(vars, s)
        end
    end

    for v in vars
        for (i, e) in enumerate(eqs)
            for s in Symbolics.get_variables(e)
                if string(s) == string(v)
                    push!(idx, i)
                end
            end
        end
    end
    return idx, vars
end

function get_idx_tagged_vars(sys, tag)
    idx = Int[]
    for (i, s) in enumerate(unknowns(sys))
        if (getdescription(s) == tag)
            push!(idx, i)
        end
    end
    return idx
end

getnontunableparams(sys) = [Symbolics.getdefaultval(p) for p ∈ parameters(sys) if !istunable(p)]
