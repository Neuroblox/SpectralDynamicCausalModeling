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
function setup_spDCM(data, model, initcond, csdsetup, priors, hyperpriors, indices, modelparam, modality)
    # compute cross-spectral density
    dt = csdsetup.dt;                      # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    freq = csdsetup.freq;                  # frequencies at which the CSD is evaluated
    mar_order = csdsetup.mar_order;        # order of MAR
    mar = mar_ml(data, mar_order);         # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freq, dt^-1);     # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    statevals = [v for v in values(initcond)]
    append!(statevals, zeros(length(unknowns(model)) - length(statevals)))
    f_model = generate_function(model; expression=Val{false})[1]
    f_at(params, t) = states -> f_model(states, MTKParameters(model, params)..., t)
    derivatives = par -> jacobian(f_at(addnontunableparams(par, model), t), statevals)

    μθ_pr = vecparam(priors.μθ_pr)        # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = priors.Σθ_pr

    ### Collect prior means and covariances ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = csd_Q(y_csd);             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)
    nr = length(indices[:lnγ])        # region specific noise parameter can be used to get the number of regions

    f = params -> csd_mtf(freq, mar_order, derivatives, params, indices, modality)

    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLMTKState(
        0,                                   # iter
        -4,                                  # log ascent rate
        [-Inf],                              # free energy
        Float64[],                           # delta free energy
        hyperpriors.μλ_pr,                   # metaparameter, initial condition.
        zeros(np),                           # parameter estimation error ϵ_θ
        [zeros(np), hyperpriors.μλ_pr],      # memorize reset state
        μθ_pr,                               # parameter posterior mean
        Σθ_pr,                               # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLMTKSetup(
        f,                                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                                # empirical cross-spectral density
        1e-1,                                 # tolerance
        [nr, np, ny, nq, nh],                 # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, hyperpriors.μλ_pr],           # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors.Πλ_pr],      # parameter and hyperparameter prior precision matrices
        Q,                                    # components of data precision matrix
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

    ### Collect prior means and covariances ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors.Q;
    else
        Q = csd_Q(y_csd);             # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)
    nr = size(priors.μθ_pr.A, 1)        # region specific noise parameter can be used to get the number of regions

    f = params -> csd_fmri_mtf(initcond, freq, mar_order, params)

    Σθ_pr, V = removezerovardims(priors.Σθ_pr)

    np = size(V, 2)               # number of parameters
    ny = length(y_csd)            # total number of response variables (dimensions x number of frequencies)


    # variational laplace state variables
    vlstate = VLState(
        0,                                   # iter
        -4,                                  # log ascent rate
        [-Inf],                              # free energy
        Float64[],                           # delta free energy
        hyperpriors.μλ_pr,                   # metaparameter, initial condition.
        zeros(np),                           # parameter estimation error ϵ_θ
        [zeros(np), hyperpriors.μλ_pr],      # memorize reset state
        priors.μθ_pr,                        # parameter posterior mean
        Σθ_pr,                        # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLSetup(
        f,                                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                                # empirical cross-spectral density
        1e-1,                                 # tolerance
        [nr, np, ny, nq, nh],                 # number of parameters, number of data points, number of Qs, number of hyperparameters
        [priors.μθ_pr, hyperpriors.μλ_pr],           # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors.Πλ_pr, V],      # parameter and hyperparameter prior precision matrices
        Q,                                    # components of data precision matrix
        priors.μθ_pr
    )
    return (vlstate, vlsetup)
end

