
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
    indices2 = (dspars = collect(1:np), u = idx_u, m = idx_bold, sts = idx_sts)

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

    # continue with prior variances
    paramvariance = copy(parammean)
    paramvariance[:lnγ] = ones(Float64, nrr)./64.0;
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
    return parammean, diagm(vecparam(paramvariance)), indices, indices2
end

### Blox Connector and Utilities ###

function generate_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    weight = get_weight(kwargs, name_out, name_in)
    if typeof(weight) == Num   # Symbol
        w = weight
    else
        w_name = Symbol("w_$(name_out)_$(name_in)")
        w = only(@parameters $(w_name)=weight [tunable=false])
    end

    return w
end

function Connector(
    blox_src::CanonicalMicroCircuitBlox,
    blox_dest::CanonicalMicroCircuitBlox;
    kwargs...
)
    sysparts_src = get_parts(blox_src)
    sysparts_dest = get_parts(blox_dest)

    wm = get_weightmatrix(kwargs, namespaced_nameof(blox_src), namespaced_nameof(blox_dest))

    idxs = findall(!iszero, wm)

    conn = mapreduce(merge!, idxs) do idx
        Connector(sysparts_src[idx[2]], sysparts_dest[idx[1]]; weight=wm[idx])
    end

    return conn
end

function Connector(
    blox_src::StimulusBlox,
    blox_dest::CanonicalMicroCircuitBlox;
    kwargs...
)
    sysparts_dest = get_parts(blox_dest)
    conn = Connector(blox_src, sysparts_dest[1]; kwargs...)

    return conn
end

function Connector(
    blox_src::CanonicalMicroCircuitBlox,
    blox_dest::ObserverBlox;
    kwargs...
)
    sysparts_src = get_parts(blox_src)
    conn = Connector(sysparts_src[2], blox_dest; kwargs...)

    return conn
end
# define a sigmoid function
# @register_symbolic sigmoid(x, r) = one(x) / (one(x) + exp(-r*x))
sigmoid(x, r) = one(x) / (one(x) + exp(-r*x)) - 0.5

function Connector(
    blox_src::JansenRitSPM, 
    blox_dest::JansenRitSPM; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    x = only(ModelingToolkit.outputs(blox_src; namespaced=true))
    r = ModelingToolkit.namespace_expr(blox_src.params[2], sys_src)

    eq = sys_dest.jcn ~ sigmoid(x, r)*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=[w, r])
end

function Connector(
    blox_src::NeuralMassBlox, 
    blox_dest::NeuralMassBlox; 
    kwargs...
)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(ModelingToolkit.outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

# additional dispatch to connect to hemodynamic observer blox
function Connector(
    blox_src::NeuralMassBlox, 
    blox_dest::ObserverBlox;
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    x = only(ModelingToolkit.outputs(blox_src; namespaced=true))
    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    eq = sys_dest.jcn ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

# additional dispatch to connect to a stimulus blox, first crafted for ExternalInput
function Connector(
    blox_src::StimulusBlox,
    blox_dest::NeuralMassBlox;
    kwargs...)

    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    w = generate_weight_param(blox_src, blox_dest; kwargs...)
    x = only(ModelingToolkit.outputs(blox_src; namespaced=true))
    eq = sys_dest.jcn ~ x*w

    return Connector(nameof(sys_src), nameof(sys_dest); equation=eq, weight=w)
end

### Namespacing Utilities ###
function get_namespaced_sys(blox)
    sys = get_system(blox)

    System(
        equations(sys),
        only(independent_variables(sys)),
        unknowns(sys),
        parameters(sys);
        name = namespaced_nameof(blox),
        discrete_events = discrete_events(sys)
    )
end

get_namespaced_sys(sys::ModelingToolkit.AbstractODESystem) = sys

import ModelingToolkit: nameof
nameof(blox) = (nameof ∘ get_system)(blox)

namespaceof(blox) = blox.namespace

namespaced_nameof(blox) = namespaced_name(inner_namespaceof(blox), nameof(blox))

"""
    Returns the complete namespace EXCLUDING the outermost (highest) level.
    This is useful for manually preparing equations (e.g. connections, see BloxConnector),
    that will later be composed and will automatically get the outermost namespace.
""" 
function inner_namespaceof(blox)
    parts = split((string ∘ namespaceof)(blox), '₊')
    if length(parts) == 1
        return nothing
    else
        return join(parts[2:end], '₊')
    end
end

namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(::Nothing, name) = Symbol(name)

function find_eq(eqs::Union{AbstractVector{<:Equation}, Equation}, lhs)
    findfirst(eqs) do eq
        lhs_vars = get_variables(eq.lhs)
        length(lhs_vars) == 1 && isequal(only(lhs_vars), lhs)
    end
end

function ModelingToolkit.outputs(blox::AbstractBlox; namespaced=false)
    sys = get_namespaced_sys(blox)
    
    return namespaced ? ModelingToolkit.namespace_expr.(ModelingToolkit.outputs(sys), Ref(sys)) : ModelingToolkit.outputs(sys)
end 

function ModelingToolkit.inputs(blox::AbstractBlox; namespaced=false)
    sys = get_namespaced_sys(blox)
    
    return namespaced ? ModelingToolkit.namespace_expr.(ModelingToolkit.inputs(sys), Ref(sys)) : ModelingToolkit.inputs(sys)
end 

ModelingToolkit.equations(blox::AbstractBlox) = ModelingToolkit.equations(get_namespaced_sys(blox))

ModelingToolkit.unknowns(blox::AbstractBlox) = ModelingToolkit.unknowns(get_namespaced_sys(blox))

ModelingToolkit.parameters(blox::AbstractBlox) = ModelingToolkit.parameters(get_namespaced_sys(blox))

get_equations_with_state_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> !ModelingToolkit.isparameter(eq.lhs), eqs)

get_equations_with_parameter_lhs(eqs::AbstractVector{<:Equation}) = filter(eq -> !ModelingToolkit.isparameter(eq.lhs), eqs)

function Base.merge!(c1::Connector, c2::Connector)
    append!(c1.source, c2.source)
    append!(c1.destination, c2.destination)
    accumulate_equations!(c1.equation, c2.equation)
    append!(c1.weight, c2.weight)
    append!(c1.discrete_callbacks, c2.discrete_callbacks)
    return c1
end

Base.merge(c1::Connector, c2::Connector) = Base.merge!(deepcopy(c1), c2)


"""
    Returns the equations for all input variables of a system, 
    assuming they have a form like : `sys.input_variable ~ ...`
    so only the input appears on the LHS.

    Input equations are namespaced by the inner namespace of blox
    and then they are returned. This way during system `compose` downstream,
    the higher-level namespaces will be added to them.

    If blox isa AbstractComponent, it is assumed that it contains a `connector` field,
    which holds a `Connector` object with all relevant connections 
    from lower levels and this level.
"""
function get_input_equations(blox::Union{AbstractBlox, ObserverBlox}; namespaced=true)
    sys = get_system(blox)
    sys_eqs = equations(sys)

    inps = ModelingToolkit.inputs(sys)
    filter!(inp -> isnothing(find_eq(sys_eqs, inp)), inps)

    if !isempty(inps)
        eqs = if namespaced
            map(inps) do inp
                ModelingToolkit.namespace_equation(
                    inp ~ 0, 
                    sys,
                    namespaced_name(inner_namespaceof(blox), nameof(blox))
                ) 
            end
        else
            map(inps) do inp
                inp ~ 0
            end
        end

        return eqs
    else
        return Equation[]
    end
end

get_input_equations(blox) = []

function accumulate_equations!(eqs::AbstractVector{<:Equation}, bloxs)
    init_eqs = mapreduce(get_input_equations, vcat, bloxs)
    accumulate_equations!(eqs, init_eqs)

    return eqs
end

function accumulate_equations!(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs1, eq)
        else
            eqs1[idx] = eqs1[idx].lhs ~ eqs1[idx].rhs + eq.rhs
        end
    end

    return eqs1
end

function accumulate_equations(eqs1::Vector{<:Equation}, eqs2::Vector{<:Equation})
    eqs = copy(eqs1)
    for eq in eqs2
        lhs = eq.lhs
        idx = find_eq(eqs1, lhs)
        
        if isnothing(idx)
            push!(eqs, eq)
        else
            eqs[idx] = eqs[idx].lhs ~ eqs[idx].rhs + eq.rhs
        end
    end

    return eqs
end

ModelingToolkit.equations(c::Connector) = c.equation

Graphs.weights(c::Connector) = c.weight

discrete_callbacks(c::Connector) = c.discrete_callbacks

function get_weight(kwargs, name_blox1, name_blox2)
    get(kwargs, :weight) do
        @info "Connection weight from $name_blox1 to $name_blox2 is not specified. Assuming weight=1" 
        return 1.0
    end
end

function get_weightmatrix(kwargs, name_blox1, name_blox2)
    get(kwargs, :weightmatrix) do
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

### State and Parameter Utilities ###

"""
    Helper to aggregate weights into a single vector
"""
function params(bc::Connector)
    wt = map(weights(bc)) do w
        Symbolics.get_variables(w)
    end

    return reduce(vcat, wt)
end


"""
    function paramscoping(;tunable=true, kwargs...)
    
    Scope arguments that are already a symbolic model parameter thereby keep the correct namespace 
    and make those that are not yet symbolic a symbol.
    Keyword arguments are used, because parameter definition require names, not just values.
"""
function paramscoping(;tunable=true, kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Num
            paramlist = vcat(paramlist, ParentScope(v))
        else
            paramlist = vcat(paramlist, @parameters $kw = v [tunable=tunable])
        end
    end
    return paramlist
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


"""
    function addnontunableparams(param, model)
    
    Function adds parameters of a model that were not marked as tunable to a list of tunable parameters
    and respects the MTK ordering of parameters.

    Arguments:
    - `paramlist`: parameters of an MTK system that were tagged as tunable
    - `sys`: MTK system

    Returns:
    - `completeparamlist`: complete parameter list of a system, including those that were not tagged as tunable
"""
function addnontunableparams(paramlist::Vector{T}, sys) where T
    completeparamlist = Vector{T}()
    k = 0
    for p in parameters(sys)
        if istunable(p)
            k += 1
            push!(completeparamlist, paramlist[k])
        else
            push!(completeparamlist, Symbolics.getdefaultval(p))
        end
    end
    append!(completeparamlist, paramlist[k+1:end])

    return completeparamlist
end

function changetune(model, parlist)
    parstochange = keys(parlist)
    p_new = map(parameters(model)) do p
        p in parstochange ? setmetadata(p, ModelingToolkit.VariableTunable, parlist[p]) : p
    end
    System(equations(model), ModelingToolkit.get_iv(model), unknowns(model), p_new; name=model.name)
end

### Graph Utilities ###
to_vector(v::AbstractVector) = v
to_vector(v) = [v]


function Connector(
    src::Union{Symbol, Vector{Symbol}}, 
    dest::Union{Symbol, Vector{Symbol}}; 
    equation=Equation[], 
    weight=Num[], 
    discrete_callbacks=[], 
    connection_blox=Set([])
    )

    Connector(
        to_vector(src), 
        to_vector(dest), 
        to_vector(equation), 
        to_vector(weight), 
        to_vector(discrete_callbacks), 
    )
end

function Base.isempty(conn::Connector)
    return isempty(conn.equation) && isempty(conn.weight)  && isempty(conn.discrete_callbacks)
end

Base.show(io::IO, c::Connector) = print(io, "$(c.source) => $(c.destination) with ", c.equation)

function show_field(io::IO, v::AbstractVector, title)
    if !isempty(v)
        println(io, title, " :")
        for val in v
            println(io, "\t $(val)")
        end
    end
end

function show_field(io::IO, d::Dict, title)
    if !isempty(d)
        println(io, title, " :")
        for (k, v) in d
            println(io, "\t ", k, " => ", v)
        end
    end
end


get_connectors(blox::CompositeBlox) = blox.connector
get_connectors(blox) = [Connector(namespaced_nameof(blox), namespaced_nameof(blox))]

get_connector(blox::CompositeBlox) = reduce(merge!, get_connectors(blox))
get_connector(blox) = Connector(namespaced_nameof(blox), namespaced_nameof(blox))

function merge_discrete_callbacks(cbs::AbstractVector)
    cbs_functional = Pair[]
    cbs_symbolic = Pair[]

    for cb in cbs
        if last(cb) isa Tuple
            push!(cbs_functional, cb)
        else
            push!(cbs_symbolic, cb)
        end
    end

    # We need to take care of the edge case where the same condition appears multiple times 
    # with the same affect. If we merge them using unique(affects) then the affect will apply only once.
    # But it could be the case that we want this affect to apply as many times as it appears in duplicate callbacks,
    # e.g. because it is a spike affect coming from different sources that happen to spike exactly at the same condition. 
    conditions = unique(first.(cbs_symbolic))
    for c in conditions
        idxs = findall(cb -> first(cb) == c, cbs_symbolic)
        affects = mapreduce(last, vcat, cbs_symbolic[idxs])
        idxs = eachindex(affects)

        affects_to_merge = Equation[]
        for (i, aff) in enumerate(affects)
            idxs_rest = setdiff(idxs, i)
            if isnothing(findfirst(x -> aff == x, affects[idxs_rest]))
                # If the affect has no duplicate then accumulate it for merging.
                push!(affects_to_merge, aff)
            else
                # If the affect has a duplicate then add them as separate callbacks
                # so that each one triggers as intended. 
                push!(cbs_functional, c => [aff])
            end
        end
        if !isempty(affects_to_merge)
            push!(cbs_functional, c => affects_to_merge)
        end
    end
   
    return cbs_functional
end

ModelingToolkit.generate_discrete_callbacks(blox, ::Connector; t_block = missing) = []

function ModelingToolkit.generate_discrete_callbacks(bc::Connector, eqs::AbstractVector{<:Equation}; t_block = missing)
    eqs_params = get_equations_with_parameter_lhs(eqs)
    dc = discrete_callbacks(bc)

    if !ismissing(t_block) && !isempty(eqs_params)
        cb_params = (t_block - sqrt(eps(float(t_block)))) => eqs_params
        return vcat(cb_params, dc)
    else
        return dc
    end 
end

function ModelingToolkit.generate_discrete_callbacks(g::MetaDiGraph, bc::Connector, eqs::AbstractVector{<:Equation}; t_block = missing)
    bloxs = flatten_graph(g)

    cbs = mapreduce(vcat, bloxs) do blox
        ModelingToolkit.generate_discrete_callbacks(blox, bc; t_block)
    end
    cbs_merged = merge_discrete_callbacks(to_vector(cbs))

    cbs_connections = ModelingToolkit.generate_discrete_callbacks(bc, eqs; t_block)

    return vcat(cbs_merged, cbs_connections)
end


function find_blox(g::MetaDiGraph, blox)
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        b == blox && return v
    end

    return nothing
end

function Graphs.add_edge!(g::MetaDiGraph, p::Pair; kwargs...)
    src, dest = p
    
    src_idx = find_blox(g, src)
    
    if isnothing(src_idx)
        add_blox!(g, src)
        src_idx = nv(g)
    end
    
    dest_idx = find_blox(g, dest)

    if isnothing(dest_idx)
        add_blox!(g, dest)
        dest_idx = nv(g)
    end

    Graphs.add_edge!(g, src_idx, dest_idx, Dict(kwargs))
end

function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

function get_bloxs(g::MetaDiGraph)
    bs = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        push!(bs, b)
    end
    return bs
end

get_parts(blox::CompositeBlox) = blox.parts
get_parts(blox::Union{AbstractBlox, ObserverBlox}) = blox

get_components(blox::CompositeBlox) = mapreduce(get_components, vcat, get_parts(blox))
get_components(blox::Vector{<:AbstractBlox}) = mapreduce(get_components, vcat, blox)
get_components(blox) = [blox]

get_system(g::MetaDiGraph) = get_system.(get_bloxs(g))

flatten_graph(g::MetaDiGraph) = mapreduce(get_components, vcat, get_bloxs(g))

get_system(blox) = blox.system
get_system(sys::ModelingToolkit.AbstractODESystem) = sys


function connectors_from_graph(g::MetaDiGraph)
    conns = reduce(vcat, get_connectors.(get_bloxs(g)))
    for edge in edges(g)

        blox_src = get_prop(g, edge.src, :blox)
        blox_dest = get_prop(g, edge.dst, :blox)

        kwargs = props(g, edge)
        push!(conns, Connector(blox_src, blox_dest; kwargs...))
    end
   
    filter!(conn -> !isempty(conn), conns)

    return conns
end

function connector_from_graph(g::MetaDiGraph)
    conns = connectors_from_graph(g)

    return isempty(conns) ? Connector(:none, :none) : reduce(merge!, conns)
end


"""
    system_from_graph(g::MetaDiGraph, p=Num[]; name, simplify=true, graphdynamics=false, kwargs...)

Take in a `MetaDiGraph` `g` describing a network of neural structures (and optionally a vector of extra parameters `p`) and construct a `System` which can be used to construct various `Problem` types (i.e. `ODEProblem`) for use with DifferentialEquations.jl solvers.

If `simplify` is set to `true` (the default), then the resulting system will have `structural_simplify` called on it with any remaining keyword arguments forwared to `structural_simplify`. That is,
```
@named sys = system_from_graph(g; kwarg1=x, kwarg2=y)
```
is equivalent to
```
@named sys = system_from_graph(g; simplify=false)
sys = structural_simplify(sys; kwarg1=x, kwarg2=y)
```
See the docstring for `structural_simplify` for information on which options it supports.

If `graphdynamics=true` (defaults to `false`), the output will be a `GraphSystem` from [GraphDynamics.jl](https://github.com/Neuroblox/GraphDynamics.jl), and the `kwargs` will be sent to the `GraphDynamics` constructor instead of using [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl/). The GraphDynamics.jl backend is typically significantly faster for large neural systems than the default backend, but is experimental and does not yet support all Neuroblox.jl features. 
"""
function system_from_graph(g::MetaDiGraph, p::Vector{Num}=Num[]; name=nothing, t_block=missing, simplify=true, graphdynamics=false, kwargs...)
    if graphdynamics
        isempty(p) || error(ArgumentError("The GraphDynamics.jl backend does yet support extra parameter lists. Got $p."))
        GraphDynamicsInterop.graphsystem_from_graph(g; kwargs...)
    else
        if isnothing(name)
            throw(UndefKeywordError(:name))
        end
        
        conns = connectors_from_graph(g)
    
        return system_from_graph(g, conns, p; name, t_block, simplify, kwargs...)
    end
end

function system_from_graph(g::MetaDiGraph, conns::AbstractVector{<:Connector}, p::Vector{Num}=Num[]; name=nothing, t_block=missing, simplify=true, graphdynamics=false, kwargs...)
    bloxs = get_bloxs(g)
    blox_syss = get_system.(bloxs)

    bc = isempty(conns) ? Connector(name, name) : reduce(merge!, conns)

    eqs = equations(bc)
    eqs_init = mapreduce(get_input_equations, vcat, bloxs)
    accumulate_equations!(eqs_init, eqs)

    connection_eqs = get_equations_with_state_lhs(eqs_init)

    discrete_cbs = identity.(ModelingToolkit.generate_discrete_callbacks(g, bc, eqs_init; t_block))
    sys = compose(System(connection_eqs, t, [], vcat(params(bc), p); name, discrete_events = discrete_cbs), blox_syss)

    if simplify
        structural_simplify(sys; kwargs...)
    else
        sys
    end
end

function system_from_parts(parts::AbstractVector; name)
    return compose(System(Equation[], t; name), get_system.(parts))
end