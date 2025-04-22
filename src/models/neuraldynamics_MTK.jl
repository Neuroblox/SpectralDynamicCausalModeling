"""
    LinearNeuralMass(name, namespace)

Create standard linear neural mass blox with a single internal state.
There are no parameters in this blox.
This is a blox of the sort used for spectral DCM modeling.
The formal definition of this blox is:


```math
\\frac{d}{dx} = \\sum{jcn}
```

where ``jcn``` is any input to the blox.


Arguments:
- name: Options containing specification about deterministic.
- namespace: Additional namespace above name if needed for inheritance.
"""

struct LinearNeuralMass <: NeuralMassBlox
    system
    namespace

    function LinearNeuralMass(;name, namespace=nothing)
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        eqs = [D(x) ~ jcn]
        sys = System(eqs, t, name=name)
        new(sys, namespace)
    end
end

# Simple input blox
mutable struct ExternalInput <: StimulusBlox
    namespace
    system

    function ExternalInput(;name, I=0.0, namespace=nothing)
        sts = @variables u(t)=0.0 [output=true, irreducible=true, description="ext_input"]
        eqs = [u ~ I]
        odesys = System(eqs, t, sts, []; name=name)

        new(namespace, odesys)
    end
end

"""
Ornstein-Uhlenbeck process Blox

variables:
    x(t):  value
    jcn:   input 
parameters:
    τ:      relaxation time
	μ:      average value
	σ:      random noise (variance of OU process is τ*σ^2/2)
returns:
    an ODE System (but with brownian parameters)
"""
mutable struct OUBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    namespace
    stochastic
    system
    function OUBlox(;name, namespace=nothing, μ=0.0, σ=1.0, τ=1.0)
        p = paramscoping(μ=μ, τ=τ, σ=σ)
        μ, τ, σ = p
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        @brownian w

        eqs = [D(x) ~ (-x + μ + jcn)/τ + sqrt(2/τ)*σ*w]
        sys = System(eqs, t; name=name)
        new(namespace, true, sys)
    end
end

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM
"""
mutable struct JansenRitSPM <: NeuralMassBlox
    params
    system
    namespace
    function JansenRitSPM(;name, namespace=nothing, τ=1.0, r=2.0/3.0)
        p = paramscoping(τ=τ, r=r)
        τ, r = p

        sts    = @variables x(t)=0.0 [output=true] y(t)=0.0 jcn(t)=0.0 [input=true]
        eqs    = [D(x) ~ y,
                  D(y) ~ (-2*y - x/τ + jcn)/τ]

        sys = System(eqs, t, name=name)
        new(p, sys, namespace)
    end
end

# Canonical micro-circuit model
mutable struct CanonicalMicroCircuitBlox <: CompositeBlox
    namespace
    parts
    system
    connector

    function CanonicalMicroCircuitBlox(;name, namespace=nothing, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @named ss = JansenRitSPM(;namespace=namespaced_name(namespace, name), τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = JansenRitSPM(;namespace=namespaced_name(namespace, name), τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = JansenRitSPM(;namespace=namespaced_name(namespace, name), τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = JansenRitSPM(;namespace=namespaced_name(namespace, name), τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        sblox_parts = vcat(ss, sp, ii, dp)

        add_edge!(g, ss => ss; :weight => -800.0)
        add_edge!(g, sp => ss; :weight => -800.0)
        add_edge!(g, ii => ss; :weight => -1600.0)
        add_edge!(g, ss => sp; :weight =>  800.0)
        add_edge!(g, sp => sp; :weight => -800.0)
        add_edge!(g, ss => ii; :weight =>  800.0)
        add_edge!(g, ii => ii; :weight => -800.0)
        add_edge!(g, dp => ii; :weight =>  400.0)
        add_edge!(g, ii => dp; :weight => -400.0)
        add_edge!(g, dp => dp; :weight => -200.0)

        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(sblox_parts; name)

        new(namespace, sblox_parts, sys, bc)
    end
end