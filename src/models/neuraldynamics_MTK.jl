# Simple input blox
mutable struct ExternalInput <: Neuroblox.StimulusBlox
    namespace
    system

    function ExternalInput(;name, I=0.0, namespace=nothing)
        sts = @variables u(t)=0.0 [output=true, irreducible=true, description="ext_input"]
        eqs = [u ~ I]
        odesys = System(eqs, t, sts, []; name=name)

        new(namespace, odesys)
    end
end