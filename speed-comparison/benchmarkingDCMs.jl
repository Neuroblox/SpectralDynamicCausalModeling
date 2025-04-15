using MAT
using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ForwardDiff
using ComponentArrays
using MetaGraphs
using Graphs

# both of the following are needed for typedefinitions even if no symbolic procedure is used
using ModelingToolkit
using OrderedCollections

### a few packages relevant for speed tests and profiling ###
using Serialization
using StatProfilerHTML

include("../src/utils/typedefinitions.jl")
include("../src/models/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("../src/variationallaplace/transferfunction.jl")
include("../src/utils/helperfunctions.jl")
include("../src/utils/helperfunctions_AD.jl")
include("../src/variationallaplace/optimization.jl")             # switch between _spm and _AD version.
include("../src/utils/mar.jl")                       # multivariate auto-regressive model functions
include("../src/utils/spDCMsetup.jl")
include("../src/models/neuraldynamics_MTK.jl")
include("../src/models/measurement_MTK.jl")
include("../src/utils/MTK_utilities.jl")


const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits


function wrapperfunction(vars)
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
    
    return setup_spDCM(data, initcond, csdsetup, priors, hyperpriors);
end


function wrapperfunction_MTK(vars)
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

    return setup_spDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, pmean, "fMRI");
end

dx = exp(-8)
max_iter = 128
# speed comparison between different DCM implementations
for n in 2:10
    vals = matread("speed-comparison/matlab_" * string(n) * "regions.mat");

    (state, setup) = wrapperfunction(vals)
    run_spDCM_iteration!(state, setup, dx)
    (state, setup) = wrapperfunction(vals)
    t_juliaSPM = @elapsed for iter in 1:max_iter
        state.iter = iter
        run_spDCM_iteration!(state, setup, dx)
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                print("convergence after ", iter, " steps, with free energy: ", state.F[end])
                break
            end
        end
    end
    F_SPM = state.F[end]
    iter_SPM = state.iter

    (state, setup) = wrapperfunction(vals)
    run_spDCM_iteration!(state, setup)
    (state, setup) = wrapperfunction(vals)
    t_juliaAD = @elapsed for iter in 1:max_iter
        state.iter = iter
        run_spDCM_iteration!(state, setup)
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                print("convergence after ", iter, " steps, with free energy: ", state.F[end])
                break
            end
        end
    end
    F_AD = state.F[end]
    iter_AD = state.iter

    (state, setup) = wrapperfunction_MTK(vals)
    run_spDCM_iteration!(state, setup)
    (state, setup) = wrapperfunction_MTK(vals)
    t_juliaMTK = @elapsed for iter in 1:max_iter
        state.iter = iter
        run_spDCM_iteration!(state, setup)
        if iter >= 4
            criterion = state.dF[end-3:end] .< setup.tolerance
            if all(criterion)
                print("convergence after ", iter, " steps, with free energy: ", state.F[end])
                break
            end
        end
    end
    F_MTK = state.F[end]
    iter_MTK = state.iter
    @show "Iteration:", n, t_juliaAD, t_juliaSPM, t_juliaMTK

    matwrite("speedcomp" * string(n) * "regions.mat", Dict(
        "t_mat" => vals["matcomptime"],
        "F_mat" => vals["F"],
        "t_jad" => t_juliaAD,
        "F_jad" => F_AD,
        "t_jspm" => t_juliaSPM,
        "F_jspm" => F_SPM,
        "t_mtk" => t_juliaMTK,
        "F_mtk" => F_MTK,
        "iter_spm" => iter_SPM,
        "iter_ad" => iter_AD,
        "iter_mtk" => iter_MTK
    ); compress = true)    
end

error()

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
