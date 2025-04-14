@parameters t
D = Differential(t)

"""
Issue with parameters at level0, see parameterscoping.
"""
function scope_dict(para_dict::Dict{Symbol, Union{Real,Num}})
    para_dict_copy = copy(para_dict)
    for (n,v) in para_dict_copy
        if typeof(v) == Num
            para_dict_copy[n] = ParentScope(v)
        else
            para_dict_copy[n] = (@parameters $n=v)[1]
        end
    end
    return para_dict_copy
end


"""
Hemodynamic Response Function modeled by the sum of two gamma functions
A[1] and A[2] model the magnitudes of the peak and undershoot terms, respectively.
τ[1] and τ[2] model the width, peak height, and time-to-peak.
δ[1] and δ[2] model the time-to-onset.

Reference: Handwerker et al. 2004, NeuroImage
"""
function GammaHRF(;name, A=[3.5, -1.5], δ=[1.5, 7.8], τ=[4.0, 4.0], C=-0.5)
    params = @parameters A=A δ=δ τ=τ C=C
    states = @variables x(t), y(t)

    eqs = [
        y ~ A[1] * ((x - δ[1])/τ[1])^2 * exp(-((x - δ[1])/τ[1])^2)/τ[1] + 
        A[2]*((x - δ[2])/τ[2])^2 * exp(-((x - δ[2])/τ[2])^2)/τ[2] + C
    ]

    return ODESystem(eqs, t, states, params; name=name)
end

function hemodynamicsMTK(;name, κ=0.0, τ=0.0)
    #= hemodynamic parameters
        H(1) - signal decay                                   d(ds/dt)/ds)
        H(2) - autoregulation                                 d(ds/dt)/df)
        H(3) - transit time                                   (t0)
        H(4) - exponent for Fout(v)                           (alpha)
        H(5) - resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    params = scope_dict(Dict(:κ=>κ, :τ=>τ))
    κ = params[:κ]
    τ = params[:τ]
    states = @variables s(t) lnf(t) lnν(t) lnq(t) x(t)

    eqs = [
        D(s)   ~ x - H[1]*exp(κ)*s - H[2]*(exp(lnf) - 1),
        D(lnf) ~ s / exp(lnf),
        D(lnν) ~ (exp(lnf) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(τ)*exp(lnν)),
        D(lnq) ~ (exp(lnf)/exp(lnq)*((1 - (1 - H[5])^(exp(lnf)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(τ))
    ]

    return ODESystem(eqs, t, states, values(params); name=name)
end


function boldsignal(;name, lnϵ=0.0)
    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE

    params = @parameters lnϵ=lnϵ
    vars = @variables bold(t) q(t) ν(t)

    eqs = [
        bold ~ V0*(k1 - k1*exp(q) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(q)/exp(ν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(ν))
    ]

    ODESystem(eqs, t, vars, values(params); name=name)
end

function linearneuralmass(;name)
    states = @variables x(t) jcn(t)
    eqs = D(x) ~ jcn
    return ODESystem(eqs, t, states, []; name=name)
end

function linearconnectionssymbolic(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    eqs = []
    nr = length(sys)
    for i in 1:nr
       push!(eqs, sys[i].nmm.jcn ~ sum(adj_matrix[(1+(i-1)*nr):nr*i] .* connector))
    end
    return ODESystem(eqs, systems=sys; name=name)
end


function hemodynamics_jacobian(x, decay, transit)
    """
    na     - neural activity
    Components of x are:
    x[:,1] - vascular signal: s
    x[:,2] - rCBF: ln(f)
    x[:,3] - venous volume: ln(ν)
    x[:,4] - deoxyhemoglobin (dHb): ln(q)
    
    decay, transit - free parameters, set to 0 for standard parameters.
    
    This function implements the hymodynamics model (balloon model and neurovascular state eq.) described in: 
    
    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
    
    adapted from spm_fx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan
    """
    
    #= hemodynamic parameters
        H(1) - signal decay                                   d(ds/dt)/ds)
        H(2) - autoregulation                                 d(ds/dt)/df)
        H(3) - transit time                                   (t0)
        H(4) - exponent for Fout(v)                           (alpha)
        H(5) - resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    # exponentiation of hemodynamic state variables
    x[:, 2:4] = exp.(x[:, 2:4])

    # signal decay
    κ = H[1]*exp(decay)

    # transit time
    τ = H[3]*exp.(transit)

    # # Fout = f(v) - outflow
    # fv = x[:, 3].^(H[4]^-1)

    # # e = f(f) - oxygen extraction
    # ff = (1.0 .- (1.0 - H[5]).^(x[:, 2].^-1))/H[5]

    d = size(x)[1]   # number of dimensions, equals typically number of regions
    J = zeros(typeof(κ), 4d, 4d)

    J[1:d,1:d] = Matrix(-κ*I, d, d)   # TODO: make it work when κ/decay is a vector. Only solution if-clause? diagm doesn't allow scalars, [κ] would work in that case
    J[1:d,d+1:2d] = diagm(-H[2]*x[:,2])
    J[d+1:2d,1:d] = diagm( x[:,2].^-1)
    J[d+1:2d,d+1:2d] = diagm(-x[:,1]./x[:,2])
    J[2d+1:3d,d+1:2d] = diagm(x[:,2]./(τ.*x[:,3]))
    J[2d+1:3d,2d+1:3d] = diagm(-x[:,3].^(H[4]^-1 - 1)./(τ*H[4]) - (x[:,3].^-1 .*(x[:,2] - x[:,3].^(H[4]^-1)))./τ)
    J[3d+1:4d,d+1:2d] = diagm((x[:,2] .+ log(1 - H[5])*(1 - H[5]).^(x[:,2].^-1) .- x[:,2].*(1 - H[5]).^(x[:,2].^-1))./(τ.*x[:,4]*H[5]))
    J[3d+1:4d,2d+1:3d] = diagm((x[:,3].^(H[4]^-1 - 1)*(H[4] - 1))./(τ*H[4]))
    J[3d+1:4d,3d+1:4d] = diagm((x[:,2]./x[:,4]).*((1 - H[5]).^(x[:,2].^-1) .- 1)./(τ*H[5]))

    return J
end

# Simulate the BOLD signal
function boldsignal(x, lnϵ)
    """
    Simulated BOLD response to input
    FORMAT [g,dgdx] = g_fmri(x, ϵ)
    g          - BOLD response (%)
    x          - hemodynamic state vector, same as above.
    ϵ          - free parameter (note also here as above, actually ln(ϵ)), ratio of intra- to extra-vascular components

    This function implements the BOLD signal model described in: 

    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.

    adapted from spm_gx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan

    NB: Biophysical constants for 1.5T scanners:
    TE  = 0.04
    V0  = 4    
    r0  = 25
    nu0 = 40.3
    E0  = 0.4
    """

    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
    ϵ  = exp(lnϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE;
    k2  = ϵ*r0*E0*TE;
    k3  = 1 - ϵ;
    # -Output equation of BOLD signal model
    ν   = exp.(x[:,4])
    q   = exp.(x[:,5])
    bold = V0*(k1 .- k1*q .+ k2 .- k2*q./ν .+ k3 .- k3*ν)

    nd = size(x, 1)
    ∇ = zeros(typeof(k3), nd, 2nd)
    ∇[1:nd, 1:nd]     = diagm(-V0*(k3*ν .- k2*q./ν))
    ∇[1:nd, nd+1:2nd] = diagm(-V0*(k1*q .+ k2*q./ν))

    return (bold, ∇)
end