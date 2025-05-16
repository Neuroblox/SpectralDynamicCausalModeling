"""
    function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, indices)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

    Arguments:
    - `data`        : dataframe with column names corresponding to the regions of measurement.
    - `model`       : MTK model, including state evolution and measurement.
    - `initcond`    : ordered dictionary of initial conditions: make sure that the ordering corresponds to the order of the model unknowns/variables
    - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
    -- `dt`         : sampling interval
    -- `freq`       : frequencies at which to evaluate the CSD
    -- `p`          : order parameter of the multivariate autoregression model
    - `priors`      : dataframe of parameters with the following columns:
    -- `name`       : corresponds to MTK model name
    -- `mean`       : corresponds to prior mean value
    -- `variance`   : corresponds to the prior variances
    - `hyperpriors` : dataframe of parameters with the following columns:
    -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
    - `indices`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
"""
function setup_spDCM(data, model, initcond, csdsetup, priors, hyperpriors, indices, indices2, modelparam, modality)
    # compute cross-spectral density
    dt = csdsetup.dt;                      # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freq = csdsetup.freq;                  # frequencies at which the CSD is evaluated
    mar_order = csdsetup.mar_order;        # order of MAR
    if eltype(data) <: Real
        mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
        y_csd = mar2csd(mar, freq, dt^-1);     # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
    else
        y_csd = data
    end
    statevals = [v for v in values(initcond)]
    append!(statevals, zeros(length(unknowns(model)) - length(statevals)))
    f_model = generate_function(model; expression=Val{false})[1]
    f_at(params, t) = states -> f_model(states, MTKParameters(model, params)..., t)
    derivatives = par -> jacobian(f_at(addnontunableparams(par, model), t), statevals)

    μθ_pr = vecparam(priors.μθ_pr)        # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = priors.Σθ_pr

    nr = size(y_csd, 2)    # get number of regions from data
    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    ### complete hyperprior in case some fields are missing ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = [csd_Q(y_csd)];             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nh = length(Q)                   # number of precision components (this is the same as above, but may differ)
    if haskey(hyperpriors, :μλ_pr)
        μλ_pr = hyperpriors.μλ_pr;
    else
        μλ_pr = ones(nh) .* (-log(var(y_csd[:])) + 4);             # compute prior means of λ which is also the prefactor of the matrix decomposition elements Q
    end
    if haskey(hyperpriors, :Πλ_pr)
        Πλ_pr = hyperpriors.Πλ_pr;
    else
        Πλ_pr = Matrix(exp(4)*I, nh, nh)             # compute prior covariance of λ
    end

    f! = (y, params) -> csd_mtf!(y, freq, mar_order, derivatives, params, indices, indices2, modality)

    # variational laplace state variables
    vlstate = VLMTKState(
        0,                               # iter
        -4,                              # log ascent rate
        [-Inf],                          # free energy
        Float64[],                       # delta free energy
        μλ_pr,                           # metaparameter, initial condition.
        zeros(np),                       # parameter estimation error ϵ_θ
        [zeros(np), μλ_pr],              # memorize reset state
        μθ_pr,                           # parameter posterior mean
        Σθ_pr,                           # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLMTKSetup(
        f!,                               # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                           # empirical cross-spectral density
        1e-1,                            # tolerance
        [nr, np, ny, nh],                # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, μλ_pr],                  # parameter and hyperparameter prior mean
        [inv(Σθ_pr), Πλ_pr],             # parameter and hyperparameter prior precision matrices
        Q,                               # components of data precision matrix
        modelparam
    )
    return (vlstate, vlsetup)
end



"""
    function setup_sDCM(data, initcond, csdsetup, priors, hyperpriors)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    
    Arguments:
    - `data`        : dataframe with column names corresponding to the regions of measurement.
    - `initcond`    : dictionary of initial conditions, numerical values for all states
    - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
    -- `dt`         : sampling interval
    -- `freq`       : frequencies at which to evaluate the CSD
    -- `p`          : order parameter of the multivariate autoregression model
    - `priors`      : ComponentArray of parameters with the following elements:
    -- `μθ_pr`      : corresponds to prior mean value
    -- `Σθ_pr`      : corresponds to the prior variances
    - `hyperpriors` : dataframe of parameters with the following columns:
    -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
"""
function setup_spDCM(data, initcond, csdsetup, priors, hyperpriors)
    # compute cross-spectral density
    dt = csdsetup.dt;                      # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freq = csdsetup.freq;                  # frequencies at which the CSD is evaluated
    mar_order = csdsetup.mar_order;        # order of MAR
    mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freq, dt^-1);     # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    nr = size(y_csd, 2)           # get number of regions from data
    ny = length(y_csd)            # total number of response variables (dimensions x number of frequencies)
    ### complete hyperprior in case some fields are missing ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = [csd_Q(y_csd)];             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nh = length(Q)                    # number of precision components
    if haskey(hyperpriors, :μλ_pr)
        μλ_pr = hyperpriors.μλ_pr;
    else
        μλ_pr = ones(nr^2) .* (-log(var(y_csd[:])) + 4);    # compute prior means of λ which is also the prefactor of the matrix decomposition elements Q
    end

    f! = (y, params) -> csd_fmri_mtf!(y, initcond, freq, mar_order, params)

    Σθ_pr, V = removezerovardims(priors.Σθ_pr)
    np = size(V, 2)               # number of parameters

    # variational laplace state variables
    vlstate = VLState(
        0,                            # iter
        -4,                           # log ascent rate
        [-Inf],                       # free energy
        Float64[],                    # delta free energy
        μλ_pr,                        # metaparameter, initial condition.
        zeros(np),                    # parameter estimation error ϵ_θ
        [zeros(np), μλ_pr],           # memorize reset state
        priors.μθ_pr,                 # parameter posterior mean
        Σθ_pr,                        # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLSetup(
        f!,                                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                                # empirical cross-spectral density
        1e-1,                                 # tolerance
        [nr, np, ny, nh],                     # number of parameters, number of data points, number of Qs, number of hyperparameters
        [priors.μθ_pr, hyperpriors.μλ_pr],    # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors.Πλ_pr, V],   # parameter and hyperparameter prior precision matrices
        Q,                                    # components of data precision matrix
        priors.μθ_pr
    )
    return (vlstate, vlsetup)
end

