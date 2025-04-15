using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using ComponentArrays
using ModelingToolkit   # need for typedefinitions even if no symbolic procedure is used

include("src/utils/typedefinitions.jl")
include("src/models/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("src/transferfunction.jl")
include("src/utils/helperfunctions.jl")
include("src/optimization_spm25.jl")             # switch between _spm and _AD version.
include("src/utils/mar.jl")                       # multivariate auto-regressive model functions
include("src/spDCMsetup.jl")

### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("demodata/spm25_demo.mat");
max_iter = 128
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

### Compute the DCM ###
(state, setup) = setup_spDCM(data, initcond, csdsetup, priors, hyperpriors);

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