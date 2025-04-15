function diff(U, dx, f, param)
    nJ = size(U, 2)
    y0 = f(param)
    J = zeros(ComplexF64, nJ, size(y0, 1), size(y0, 2), size(y0, 3))
    for i = 1:nJ
        y1 = f(param .+ U[:, i]*dx)
        J[i,:,:,:] = (y1 .- y0)/dx
    end
    return J, y0
end

function run_spDCM_iteration!(state::VLState, setup::VLSetup, dx)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (_, np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr, V) = setup.systemmatrices
    Q = setup.Q

    dfdθ, f_μθ = diff(V, dx, f, μθ_po);
    dfdθ = transpose(reshape(dfdθ, np, ny))

    norm_dfdθ = opnorm(dfdθ, Inf);
    revert = isnan(norm_dfdθ) || norm_dfdθ > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);

            # E-Step: update
            ϵ_θ += integration_step(dFdθθ, dFdθ, v)

            μθ_po = μθ_pr + V * ϵ_θ

            dfdθ, f_μθ = diff(V, dx, f_prep, μθ_po);
            dfdθ = transpose(reshape(dfdθ, np, ny))

            # check for stability
            norm_dfdθ = opnorm(dfdθ, Inf);
            revert = isnan(norm_dfdθ) || norm_dfdθ > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f_μθ, ny)                   # error
    J = - dfdθ   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(real(eltype(J)), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        if nh > 1
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = iΣ \ P[:,:,i]
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
            end
            for i = 1:nh
                dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                    dFdλλ[j, i] = dFdλλ[i, j]
                end
            end
        else
            # if nh == 1, do the followng simplifications to improve computational speed:          
            # 1. replace trace(PΣ[1]) * nq by ny
            # 2. replace JPJ[1] by Pp
            dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;

            # 3. replace trace(PΣ[1],PΣ[1]) * nq by ny
            dFdλλ[1, 1] = - ny/2;
        end

        dFdλλ = dFdλλ + diagm(dFdλ);      # add second order terms; noting diΣ/dλ(i)dλ(i) = diΣ/dλ(i) = P{i}

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        # E-Step: update
        dλ = real(integration_step(dFdλλ, dFdλ, 4))

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    dθ = integration_step(dFdθθ, dFdθ, v, true)

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + V * ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end

"""
    variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)

    Computes parameter estimation using variational Laplace that is to a large extend equivalent to the SPM12 implementation
    and provides the exact same values.

    Arguments:
    - `idx_A`: indices of connection weight parameter matrix A in model Jacobian
    - `y`: empirical cross-spectral density (input data)
    - `derivatives`: jacobian of model as well as gradient of observer function
    - `w`: fequencies at which to estimate cross-spectral densities
    - `V`: projection matrix from full parameter space to reduced space that removes parameters with zero variance prior
    - `p`: order of multivariate autoregressive model for estimation of cross-spectral densities from data
    - `priors`: Bayesian priors, mean and variance thereof. Laplace approximation assumes Gaussian distributions
    - `niter`: number of iterations of the optimization procedure
"""
function run_spDCM_iteration!(state::VLState, setup::VLSetup)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (_, _, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr, V) = setup.systemmatrices
    Q = setup.Q

    dfdp = ForwardDiff.jacobian(f, μθ_po) * V

    norm_dfdp = opnorm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);

            # E-Step: update
            ϵ_θ += integration_step(dFdθθ, dFdθ, v)

            μθ_po = μθ_pr + V * ϵ_θ

            dfdp = ForwardDiff.jacobian(f, μθ_po) * V

            # check for stability
            norm_dfdp = opnorm(dfdp, Inf);
            revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f(μθ_po), ny)                   # error
    J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(real(eltype(J)), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        if nh > 1
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = iΣ \ P[:,:,i]
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
            end
            for i = 1:nh
                dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                    dFdλλ[j, i] = dFdλλ[i, j]
                end
            end
        else
            # if nh == 1, do the followng simplifications to improve computational speed:          
            # 1. replace trace(PΣ[1]) * nq by ny
            # 2. replace JPJ[1] by Pp
            dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;

            # 3. replace trace(PΣ[1],PΣ[1]) * nq by ny
            dFdλλ[1, 1] = - ny/2;
        end

        dFdλλ = dFdλλ + diagm(dFdλ);      # add second order terms; noting diΣ/dλ(i)dλ(i) = diΣ/dλ(i) = P{i}

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        # E-Step: update
        dλ = real(integration_step(dFdλλ, dFdλ, 4))

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    dθ = integration_step(dFdθθ, dFdθ, v, true)

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + V * ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end

"""
    variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)

    Computes parameter estimation using variational Laplace that is to a large extend equivalent to the SPM12 implementation
    and provides the exact same values.

    Arguments:
    - `idx_A`: indices of connection weight parameter matrix A in model Jacobian
    - `y`: empirical cross-spectral density (input data)
    - `derivatives`: jacobian of model as well as gradient of observer function
    - `w`: fequencies at which to estimate cross-spectral densities
    - `V`: projection matrix from full parameter space to reduced space that removes parameters with zero variance prior
    - `p`: order of multivariate autoregressive model for estimation of cross-spectral densities from data
    - `priors`: Bayesian priors, mean and variance thereof. Laplace approximation assumes Gaussian distributions
    - `niter`: number of iterations of the optimization procedure
"""
function run_spDCM_iteration!(state::VLMTKState, setup::VLMTKSetup)
    (;μθ_po, λ, v, ϵ_θ, dFdθ, dFdθθ) = state

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (nr, np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)

    norm_dfdp = opnorm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);

            # E-Step: update
            ϵ_θ += integration_step(dFdθθ, dFdθ, v)

            μθ_po = μθ_pr + ϵ_θ

            dfdp = ForwardDiff.jacobian(f, μθ_po)

            # check for stability
            norm_dfdp = opnorm(dfdp, Inf);
            revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f(μθ_po), ny)                   # error
    J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(real(eltype(J)), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        if nh > 1
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = iΣ \ P[:,:,i]
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
            end
            for i = 1:nh
                dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                    dFdλλ[j, i] = dFdλλ[i, j]
                end
            end
        else
            # if nh == 1, do the followng simplifications to improve computational speed:          
            # 1. replace trace(PΣ[1]) * nq by ny
            # 2. replace JPJ[1] by Pp
            dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;

            # 3. replace trace(PΣ[1],PΣ[1]) * nq by ny
            dFdλλ[1, 1] = - ny/2;
        end

        dFdλλ = dFdλλ + diagm(dFdλ);      # add second order terms; noting diΣ/dλ(i)dλ(i) = diΣ/dλ(i) = P{i}

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        # E-Step: update
        dλ = real(integration_step(dFdλλ, dFdλ, 4))

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    dθ = integration_step(dFdθθ, dFdθ, v, true)

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end