### type definitions ###
abstract type AbstractBlox end # Blox is the abstract type for Blox that are displayed in the GUI
abstract type AbstractComponent end
abstract type BloxConnection end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type AbstractNeuronBlox <: AbstractBlox end
abstract type NeuralMassBlox <: AbstractBlox end
abstract type CompositeBlox <: AbstractBlox end
abstract type StimulusBlox <: AbstractBlox end
abstract type ObserverBlox end   # not AbstractBlox since it should not show up in the GUI

struct Connector
    source::Vector{Symbol}
    destination::Vector{Symbol}
    equation::Vector{Equation}
    weight::Vector{Num}
    discrete_callbacks
end

# struct types for Variational Laplace
mutable struct VLState
    iter::Int                    # number of iteration
    v::Float64                   # log ascent rate of SPM style Levenberg-Marquardt optimization
    F::Vector{Float64}           # free energy vector (store at each iteration)
    dF::Vector{Float64}          # predicted free energy changes (store at each iteration)
    λ::Vector{Float64}           # hyperparameter
    ϵ_θ::Vector{Float64}         # prediction error of parameters θ
    reset_state::Vector{Any}     # store state to reset to [ϵ_θ and λ] when the free energy gets worse rather than better
    μθ_po::ComponentVector{Float64}       # posterior expectation value of parameters 
    Σθ_po::Matrix{Float64}       # posterior covariance matrix of parameters
    dFdθ::Vector{Float64}        # free energy gradient w.r.t. parameters
    dFdθθ::Matrix{Float64}       # free energy Hessian w.r.t. parameters
end

struct VLSetup{Model, T1 <: Array{ComplexF64}, T2 <: AbstractArray}
    model_at_x0::Model                        # model evaluated at initial conditions
    y_csd::T1                                 # cross-spectral density approximated by fitting MARs to data
    tolerance::Float64                        # convergence criterion
    systemnums::Vector{Int}                   # several integers -> np: n. parameters, ny: n. datapoints, nq: n. Q matrices, nh: n. hyperparameters
    systemvecs::Vector{DenseVector{Float64}}       # μθ_pr: prior expectation values of parameters and μλ_pr: prior expectation values of hyperparameters
    systemmatrices::Vector{Matrix{Float64}}   # Πθ_pr: prior precision matrix of parameters, Πλ_pr: prior precision matrix of hyperparameters
    Q::T2                                     # linear decomposition of precision matrix of parameters, typically just one matrix, the empirical correlation matrix
    modelparam::ComponentVector
end

mutable struct VLMTKState
    iter::Int                    # number of iteration
    v::Float64                   # log ascent rate of SPM style Levenberg-Marquardt optimization
    F::Vector{Float64}           # free energy vector (store at each iteration)
    dF::Vector{Float64}          # predicted free energy changes (store at each iteration)
    λ::Vector{Float64}           # hyperparameter
    ϵ_θ::Vector{Float64}         # prediction error of parameters θ
    reset_state::Vector{Any}     # store state to reset to [ϵ_θ and λ] when the free energy gets worse rather than better
    μθ_po::Vector{Float64}       # posterior expectation value of parameters 
    Σθ_po::Matrix{Float64}       # posterior covariance matrix of parameters
    dFdθ::Vector{Float64}        # free energy gradient w.r.t. parameters
    dFdθθ::Matrix{Float64}       # free energy Hessian w.r.t. parameters
end

struct VLMTKSetup{Model, T1 <: Array{ComplexF64}, T2 <: AbstractArray}
    model_at_x0::Model                        # model evaluated at initial conditions
    y_csd::T1                                 # cross-spectral density approximated by fitting MARs to data
    tolerance::Float64                        # convergence criterion
    systemnums::Vector{Int}                   # several integers -> np: n. parameters, ny: n. datapoints, nq: n. Q matrices, nh: n. hyperparameters
    systemvecs::Vector{Vector{Float64}}       # μθ_pr: prior expectation values of parameters and μλ_pr: prior expectation values of hyperparameters
    systemmatrices::Vector{Matrix{Float64}}   # Πθ_pr: prior precision matrix of parameters, Πλ_pr: prior precision matrix of hyperparameters
    Q::T2                                     # linear decomposition of precision matrix of parameters, typically just one matrix, the empirical correlation matrix
    modelparam::OrderedDict
end