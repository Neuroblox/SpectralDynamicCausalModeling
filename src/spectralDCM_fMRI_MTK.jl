using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ForwardDiff
using OrderedCollections
using ComponentArrays
using MetaGraphs
using Graphs
using ModelingToolkit
using MAT

include("utils/typedefinitions.jl")
include("utils/helperfunctions.jl")
include("utils/helperfunctions_AD.jl")
include("utils/spDCMsetup.jl")
include("variationallaplace/transferfunction.jl")
include("variationallaplace/optimization.jl")
include("utils/mar.jl")
include("models/neuraldynamics_MTK.jl")
include("models/measurement_MTK.jl")
include("utils/MTK_utilities.jl")

const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits

### Load data ###
vars = matread("demodata/spm25_demo.mat");
data = vars["data"];
nr = size(data, 2);                  # number of regions
max_iter = 128                       # maximum number of iterations

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

(state, setup) = setup_spDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, pmean, "fMRI");
for iter in 1:max_iter
    state.iter = iter
    run_spDCM_iteration!(state, setup)
    print("iteration: ", iter, " - F:", state.F[end] - state.F[2], " - dF predicted:", state.dF[end], "\n")
    if iter >= 4
        criterion = state.dF[end-3:end] .< setup.tolerance
        if all(criterion)
            print("convergence, with free energy: ", state.F[end])
            break
        end
    end
end