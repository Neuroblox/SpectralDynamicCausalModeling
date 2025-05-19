using MAT
using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ForwardDiff
using ComponentArrays
using MetaGraphs
using Graphs
using Statistics
using SparseArrays

# both of the following are needed for typedefinitions even if no symbolic procedure is used
using ModelingToolkit
using OrderedCollections

### a few packages relevant for speed tests and profiling ###
using Serialization
using StatProfilerHTML

include("../src/utils/typedefinitions.jl")
include("../src/models/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("../src/utils/helperfunctions.jl")
include("../src/utils/helperfunctions_AD.jl")
include("../src/variationallaplace/transferfunction.jl")
include("../src/variationallaplace/optimization.jl")             # switch between _spm and _AD version.
include("../src/utils/mar.jl")                       # multivariate auto-regressive model functions
include("../src/utils/spDCMsetup.jl")
include("../src/models/neuraldynamics_MTK.jl")
include("../src/models/measurement_MTK.jl")
include("../src/utils/MTK_utilities.jl")


const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits


function wrapperfunction(vars; max_iter=128, dx=nothing)
    data = vars["data"];
    nr = size(data, 2);
    
    ### Define priors and initial conditions ###
    initcond = vars["x"];                       # initial condition of dynamic variabls
    
    priors = ComponentArray(μθ_pr = (
                                        A = vars["pE"]["A"],              # prior mean of connectivity matrix
                                        C = ones(Float64, nr),            # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM...
                                        lnτ = vec(vars["pE"]["transit"]), # hemodynamic transit parameter
                                        lnκ = vars["pE"]["decay"],        # hemodynamic decay time
                                        lnϵ = vars["pE"]["epsilon"],      # BOLD signal ratio between intra- and extravascular signal
                                        lnα = vars["pE"]["a"],            # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014
                                        lnβ = vars["pE"]["b"],            # global observation noise, ln(β) as above
                                        lnγ = vec(vars["pE"]["c"])        # region specific observation noise
                                    ),
                            Σθ_pr = vars["pC"];                           # prior covariance of parameter values
                            );
    
    hyperpriors = (Πλ_pr = 128.0*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   μλ_pr = [8.0]               # prior metaparameter mean, needs to be a vector
                  );
    
    csdsetup = (mar_order = 8, freq = vec(vars["Hz"]), dt = vars["dt"]);
    (state, setup) = setup_spDCM(data, initcond, csdsetup, priors, hyperpriors);

    for iter in 1:max_iter
        state.iter = iter
        if dx === nothing
            run_spDCM_iteration!(state, setup)
        else
            run_spDCM_iteration!(state, setup, dx)
        end
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                println("convergence after ", iter, " steps, with free energy: ", state.F[end])
                break
            end
        end
    end

    return state
end


function wrapperfunction_MTK(vars; max_iter=128)
    data = vars["data"];
    nr = size(data, 2);

    ########## assemble the model ##########
    g = MetaDiGraph()
    regions = Dict()

    # decay parameter for hemodynamics lnκ and ratio of intra- to extra-vascular components lnϵ is shared across brain regions 
    @parameters lnκ=vars["pE"]["decay"] [tunable = true] lnϵ=vars["pE"]["epsilon"] [tunable=true] C=1/16 [tunable = false]   # setting tunable=true means that these parameters are optimized
    for ii = 1:nr
        region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
        add_blox!(g, region)
        regions[ii] = nv(g)    # store index of neural mass model
        taskinput = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
        add_edge!(g, taskinput => region, weight = C)
        # add hemodynamic observer
        measurement = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=lnκ, lnϵ=lnϵ, lnτ=vars["pE"]["transit"][ii])
        # connect observer with neuronal signal
        add_edge!(g, region => measurement, weight = 1.0)
    end

    # add symbolic weights
    A = []
    for (i, a) in enumerate(vec(vars["pE"]["A"]))
        symb = Symbol("A$(i)")
        push!(A, only(@parameters $symb = a))
    end

    for (i, idx) in enumerate(CartesianIndices(vars["pE"]["A"]))
        if idx[1] == idx[2]
            add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, -exp(A[i])/2)    # treatement of diagonal elements in SPM, to avoid instabilities of the linear model: see Gershgorin Circle Theorem
        else
            add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
        end
    end

    # compose model
    @named fullmodel = system_from_graph(g, simplify=false)
    untunelist = Dict()
    for (i, v) in enumerate(diag(vars["pC"])[1:nr^2])
        untunelist[A[i]] = v == 0 ? false : true
    end
    fullmodel = changetune(fullmodel, untunelist)
    fullmodel = structural_simplify(fullmodel)

    # attribute initial conditions to states
    sts, _ = get_dynamic_states(fullmodel)
    initcond = OrderedDict(sts .=> 0.0)
    rnames = []
    map(x->push!(rnames, split(string(x), "₊")[1]), sts);
    rnames = unique(rnames);
    for (i, r) in enumerate(rnames)
        for (j, s) in enumerate(sts[r .== map(x -> x[1], split.(string.(sts), "₊"))])
            initcond[s] = vars["x"][i, j]
        end
    end

    pmean, pcovariance, indices = defaultprior(fullmodel, nr)

    priors = (μθ_pr = pmean,
            Σθ_pr = pcovariance
    );
    hyperpriors = (Πλ_pr = 128.0*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                μλ_pr = [8.0]               # prior metaparameter mean, needs to be a vector
                );

    csdsetup = (mar_order = 8, freq = vec(vars["Hz"]), dt = vars["dt"]);

    (state, setup) =  setup_spDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, pmean, "fMRI");
    for iter in 1:max_iter
        state.iter = iter
        run_spDCM_iteration!(state, setup)
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                println("convergence after ", iter, " steps, with free energy: ", state.F[end])
                break
            end
        end
    end
    return state
end

function wrapperfunction_MTK_lfp(vars; max_iter=128)
    data = vars["csd"];
    nr = size(data, 2);
    dt = vars["dt"];
    freqs = vec(vars["Hz"]);

    ### Define priors and initial conditions ###
    x = vars["x"];                       # initial condition of dynamic variabls

    ########## assemble the model ##########
    g = MetaDiGraph()
    global_ns = :g # global namespace
    regions = Dict()

    @parameters lnr = 0.0
    @parameters lnτ_ss=0 lnτ_sp=0 lnτ_ii=0 lnτ_dp=0
    @parameters C=512.0 [tunable = false]    # TODO: SPM has this seemingly arbitrary 512 pre-factor in spm_fx_cmc.m. Can we understand why?
    for ii = 1:nr
        region = CanonicalMicroCircuitBlox(;namespace=global_ns, name=Symbol("r$(ii)₊cmc"), 
                                            τ_ss=exp(lnτ_ss)*0.002, τ_sp=exp(lnτ_sp)*0.002, τ_ii=exp(lnτ_ii)*0.016, τ_dp=exp(lnτ_dp)*0.028, 
                                            r_ss=exp(lnr)*2.0/3, r_sp=exp(lnr)*2.0/3, r_ii=exp(lnr)*2.0/3, r_dp=exp(lnr)*2.0/3)
        add_blox!(g, region)
        regions[ii] = nv(g)    # store index of neural mass model
        taskinput = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
        add_edge!(g, taskinput => region, weight = C)

        # add lead field (LFP measurement)
        measurement = LeadField(;name=Symbol("r$(ii)₊lf"))
        # connect measurement with neuronal signal
        add_edge!(g, region => measurement, weight = 1.0)
    end

    nl = Int((nr^2-nr)/2)   # number of links unidirectional
    @parameters a_sp_ss[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> ss: sim value 1/2
    @parameters a_sp_dp[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> dp: sim value 3/2
    @parameters a_dp_sp[1:nl] = repeat([0.0], nl)  # backward connection parameter dp -> sp: sim value 1/16
    @parameters a_dp_ii[1:nl] = repeat([0.0], nl) # backward connection parameters dp -> ii: sim value 3

    k = 0
    for i in 1:nr
        for j in (i+1):nr
            k += 1
            # forward connection matrix
            add_edge!(g, regions[i], regions[j], :weightmatrix,
                    [0 exp(a_sp_ss[k]) 0 0;            # connection from sp to ss
                    0 0 0 0;
                    0 0 0 0;
                    0 exp(a_sp_dp[k])/2 0 0] * 200)    # connection from sp to dp
            # backward connection matrix
            add_edge!(g, regions[j], regions[i], :weightmatrix,
                    [0 0 0 0;
                    0 0 0 -exp(a_dp_sp[k]);            # connection from dp to sp
                    0 0 0 -exp(a_dp_ii[k])/2;          # connection from dp to ii
                    0 0 0 0] * 200)
        end
    end

    @named fullmodel = system_from_graph(g)

    # attribute initial conditions to states
    sts, idx_sts = get_dynamic_states(fullmodel)
    idx_u = get_idx_tagged_vars(fullmodel, "ext_input")                # get index of external input state
    idx_measurement, _ = get_eqidx_tagged_vars(fullmodel, "measurement")  # get index of equation of bold state
    initcond = OrderedDict(sts .=> 0.0)
    rnames = []
    map(x->push!(rnames, split(string(x), "₊")[1]), sts);
    rnames = unique(rnames);
    for (i, r) in enumerate(rnames)
        for (j, s) in enumerate(sts[r .== map(x -> x[1], split.(string.(sts), "₊"))])
            initcond[s] = x[i, j]
        end
    end

    modelparam = OrderedDict()
    np = sum(tunable_parameters(fullmodel); init=0) do par
        val = Symbolics.getdefaultval(par)
        modelparam[par] = val
        length(val)
    end
    indices = Dict(:dspars => collect(1:np))
    # Noise parameter mean
    modelparam[:lnα] = zeros(Float64, 2, nr);         # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
    n = length(modelparam[:lnα]);
    indices[:lnα] = collect(np+1:np+n);
    np += n;
    modelparam[:lnβ] = [-16.0, -16.0];                # global observation noise, ln(β) as above
    n = length(modelparam[:lnβ]);
    indices[:lnβ] = collect(np+1:np+n);
    np += n;
    modelparam[:lnγ] = (-16.0)*ones(Float64, 2, nr);  # region specific observation noise
    n = length(modelparam[:lnγ]);
    indices[:lnγ] = collect(np+1:np+n);
    np += n
    indices[:u] = idx_u
    indices[:m] = idx_measurement
    indices[:sts] = idx_sts

    # define prior variances
    paramvariance = copy(modelparam)
    paramvariance[:lnα] = ones(Float64, size(modelparam[:lnα]))./128.0; 
    paramvariance[:lnβ] = ones(Float64, 2)./128.0;
    paramvariance[:lnγ] = ones(Float64, size(modelparam[:lnγ]))./128.0;
    for (k, v) in paramvariance
        if occursin("a_", string(k))
            paramvariance[k] = 1/16.0
        elseif "lnr" == string(k)
            paramvariance[k] = 1/64.0;
        elseif occursin("lnτ", string(k))
            paramvariance[k] = 1/32.0;
        elseif occursin("lf₊L", string(k))
            paramvariance[k] = 64;
        end
    end

    priors = (μθ_pr = modelparam,
            Σθ_pr = diagm(vecparam(paramvariance))
    );

    csdsetup = (mar_order = 8, freq = vec(vars["Hz"]), dt = vars["dt"]);
    nf = length(csdsetup.freq)
    Q = [spzeros(nf*nr^2, nf*nr^2) for i = 1:nr^2];
    for (i, q) in enumerate(Q)
        q[diagind(q)[((i-1)*nf+1):(nf*i)]] .= 1.0
    end
    
    hyperpriors = (
                    Q = Q,      # prior metaparameter precision, needs to be a matrix
                  );

    (state, setup) = setup_spDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, modelparam, "LFP");
    for iter in 1:max_iter
        state.iter = iter
        run_spDCM_iteration!(state, setup)
        print("iteration: ", iter, " - F:", state.F[end] - state.F[2], " - dF predicted:", state.dF[end], "\n")
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                print("convergence\n")
                break
            end
        end
    end
    return state
end

BLAS.set_num_threads(1)
BLAS.get_num_threads()

# speed comparison between different DCM implementations
# for n in 2:10
#     vals = matread("speed-comparison/fastspeed/matlab_" * string(n) * "regions.mat");

#     wrapperfunction(vals, max_iter=1, dx=exp(-8))
#     t_juliaSPM = @elapsed spm_state = wrapperfunction(vals, dx=exp(-8))

#     wrapperfunction(vals, max_iter=1)
#     t_juliaAD = @elapsed ad_state = wrapperfunction(vals)

#     wrapperfunction_MTK(vals, max_iter=1)
#     t_juliaMTK = @elapsed mtk_state = wrapperfunction_MTK(vals)
#     @show "Iteration:", n, t_juliaAD, t_juliaSPM, t_juliaMTK

#     matwrite("speedcomp" * string(n) * "regions.mat", Dict(
#         "t_mat" => vals["matcomptime"],
#         "F_mat" => vals["F"],
#         "t_jad" => t_juliaAD,
#         "F_jad" => ad_state.F[end],
#         "t_jspm" => t_juliaSPM,
#         "F_jspm" => spm_state.F[end],
#         "t_mtk" => t_juliaMTK,
#         "F_mtk" => mtk_state.F[end],
#         "iter_spm" => spm_state.iter,
#         "iter_ad" => ad_state.iter,
#         "iter_mtk" => mtk_state.iter
#     ); compress = true)    
# end



for n in 2:10
    vals = matread("speed-comparison/cmc_" * string(n) * "regions.mat");

    wrapperfunction_MTK_lfp(vals, max_iter=1)
    t_julialfp = @elapsed mtk_state = wrapperfunction_MTK_lfp(vals)
    @show "Iteration:", n, t_julialfp

    matwrite("speed-comparison/speedcomp_lfp_improved_sparse_" * string(n) * "regions.mat", Dict(
        "t_mat" => vals["matcomptime"],
        "F_mat" => vals["F"],
        "t_mtk" => t_julialfp,
        "F_mtk" => mtk_state.F[end],
        "iter_mtk" => mtk_state.iter
    ); compress = true)    
end


### Profiling ###
# include("../src/VariationalBayes_AD.jl")

# n = 3
# vars = matread("speedandaccuracy/matlab0.01_" * string(n) *"regions.mat");
# y = vars["data"];
# dt = vars["dt"];
# freqs = vec(vars["Hz"]);
# p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
# mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
# y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

# ### Define priors and initial conditions ###
# x = vars["x"];                       # initial condition of dynamic variabls
# A = vars["pE"]["A"];                 # initial values of connectivity matrix
# θΣ = vars["pC"];                     # prior covariance of parameter values 
# λμ = vec(vars["hE"]);                # prior mean of hyperparameters
# Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
# if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
#     Πλ_p *= ones(1,1)
# end

# # depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
# # Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
# # to what extend this is legitimate. 
# idx = findall(x -> x != 0, θΣ);
# V = zeros(size(θΣ, 1), length(idx));
# order = sortperm(θΣ[idx], rev=true);
# idx = idx[order];
# for i = 1:length(idx)
#     V[idx[i][1], i] = 1.0
# end
# θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0
# Πθ_p = inv(θΣ);

# # define a few more initial values of parameters of the model
# dim = size(A, 1);
# C = zeros(Float64, dim);          # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another little strange thing of SPM12...
# lnα = [0.0, 0.0];                 # ln(α) as in equation 2 
# lnβ = [0.0, 0.0];                 # ln(β) as in equation 2
# lnγ = zeros(Float64, dim);        # region specific observation noise parameter
# lnϵ = 0.0;                        # BOLD signal parameter
# lndecay = 0.0;                    # hemodynamic parameter
# lntransit = zeros(Float64, dim);  # hemodynamic parameters
# param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; lnα[1]; lnβ[1]; lnα[2]; lnβ[2]; lnγ;];
# # Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
# Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
# priors = [Πθ_p, Πλ_p, λμ, Q];
# variationalbayes(x, y_csd, freqs, V, param, priors, 26)

# @profilehtml results = variationalbayes(x, y_csd, freqs, V, param, priors, 26)
