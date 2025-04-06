# questions:
# 1. what is the right approach to use variables outside of for loop that are defined inside it?
# 2. is it better to divide or do power? ones(rank_X,1)./dx(1:rank_X)
# 3. Why do we also include the dimensionality here? has something to do with svd?

"""
Bayesian Multivariate Autoregressive Model estimation
mar = estimate_mar(X, p, prior)
 
Matrix of AR coefficients are in form
x_t = -a_1 x_t-1 - a_2 x_t-2 + ...... - a_p x_t-p
where a_k is a d-by-d matrix of coefficients at lag k and x_t-k's are 
vectors of a d-variate time series.
X              T-by-d matrix containing d-variate time series
p              Order of MAR model
prior          Prior on MAR coefficients (see marprior.m)
 
mar.lag(k).a   AR coefficient matrix at lag k
mar.noise_cov  Estimated noise covariance
mar.fm         Free energy of model
mar.wmean      MAR coefficients stored in a matrix
y              Target values
y_pred         Predicted values
"""
function estimate_mar(X, p, prior)

    nd = size(X,2);    # dimension of time series
    ns = size(X,1);    # length of time series

    # Embedding of multiple time series 
    # giving x=[(x1(t-1) x2(t-1) .. xd(t-1)) (x1(t-2) x2(t-2)..xd(t-2)) ...
    #           (x1(t-p) x2(t-p) .. xd(t-p))] on each row

    # TODO: offset is ignored at current, include that additional constant dimension?
    x = zeros(ns-p, nd*p)
    for i = p:-1:1
        x[:, (p-i)*nd+1:(p-i+1)*nd] = X[i:ns+i-p-1, :]
    end

    y = X[p+1:ns,:];   # target variable

    k = p*nd^2;
    # Get both pseudo-inverse and approx to inv(x'*x) efficiently
    ux, dx, vx = svd(x);
    kk=size(x, 2);
    svd_tol=maximum(dx)*eps()*kk;   # Why do we also include the dimensionality here? has something to do with svd?
    rank_X=sum(dx .> svd_tol);
    dxm=diagm(dx[1:rank_X].^-1);    # is it better to divide or do power? ones(rank_X,1)./dx(1:rank_X)
    dxm2=diagm(dx[1:rank_X].^-2);
    xp=vx[:,1:rank_X]*dxm*ux[:,1:rank_X]';   # Pseudo-inverse
    inv_xtx=vx[:,1:rank_X]*dxm2*vx[:,1:rank_X]';  # approx to inv(x'*x)

    # Compute term that will be used many times
    xtx=x'*x;

    # Get maximum likelihood solution
    # w_ml = pinv(-1*x)*y;
    w_ml = -xp*y;

    # Swap signs to be consistent with paper (swap back at end !)
    w_ml *= -1;

    y_pred = x * w_ml;
    e = y - y_pred;
    noise_cov = (e' * e)/(ns-p);
    sigma_ml = kron(noise_cov, inv_xtx);

    # Priors on alpha(s)
    b_alpha_prior=1000;
    c_alpha_prior=0.001;

    # Initialise 
    w_mean=w_ml;
    w_cov=sigma_ml;

    max_iters=32;     # TODO: put this and perhaps other algorithm parameters into the function interface
    w=zeros(p*nd, nd);
    tol=0.0001;
    for it = 1:max_iters
        
        # Update weight precisions
        Ij = diagm(prior);
        kj = sum(prior);
        E_wj = 0.5*w_mean[:]'*Ij*w_mean[:];
        b_alpha = E_wj+0.5*tr(Ij*w_cov*Ij) + (1/b_alpha_prior);
        b_alpha = 1/b_alpha;
        c_alpha = 0.5*kj+c_alpha_prior;
        mean_alpha = b_alpha*c_alpha;
        
        yy_pred = x*w_mean;
        ee = y-yy_pred;
        E_d_av = ee'*ee;
        
        Omega = zeros(nd, nd);
        # Submatrix size
        s=p*nd;
        for i = 1:nd
            for j = i:nd
                istart = 1+(i-1)*s;
                istop = istart+s-1;
                jstart = 1+(j-1)*s;
                jstop = jstart+s-1;
                w_cov_ij = w_cov[istart:istop, jstart:jstop];
                Omega[i, j] = tr(w_cov_ij*xtx);
            end
        end
        Omega = Omega+Omega'-diagm(diag(Omega));

        E_d_av .+= Omega;

        # Update noise precision posterior
        B = E_d_av;
        a = ns;
        mean_lambda = a*inv(B);

        prior_cov = zeros(k, k);
        Ij = diagm(prior);
        prior_cov = prior_cov .+ (1/mean_alpha)*Ij;

        # Convergence criterion
        old_w=w;
        w=w_mean;

        if it <= 2
            w=w_mean;
        else
            change = norm(w[:] - old_w[:])/k;
            if change < tol
                break;
            end
        end;

        # Update weight posterior
        data_precision = kron(mean_lambda, xtx);
        prior_prec = zeros(k, k);

        Ij = diagm(prior);
        prior_prec = prior_prec+mean_alpha*Ij;

        w_cov = inv(data_precision + prior_prec);
        vec_w_mean = w_cov*data_precision*w_ml[:];
        w_mean = reshape(vec_w_mean, p*nd, nd);
    end


    # Load up returning data structure
    mar = Dict()
    mar["p"] = p;

    # This is the ML estimate
    mar["noise_cov"] = noise_cov;
    mar["a_ml"] = w_ml;
    mar["sigma_ml"] = sigma_ml;
    mar["w_cov"] = w_cov;
    mar["prior"] = prior;
    # mar["mean_lambda"] = mean_lambda;
    # mar["bic"] = -0.5*N*logdet(B) - 0.5*k*log(ns);
    mar["a_post"] = zeros(nd, nd, p);
    for i = 1:p
        start = (i-1)*nd+1;
        stop = (i-1)*nd+1+(nd-1);
        # Transpose and swap signs for compatibility with spectral estimation function
        mar["a_post"][:, :, i] = -w_mean[start:stop,:]';
    end
    # Swap signs for compatibility with spectral estimation function
    mar["wmean"] = -w_mean;
    mar["nd"] = nd;

    return mar
end


"""
Maximum likelihood estimator of a multivariate, or vector auto-regressive model.
    y : MxN Data matrix where M is number of samples and N is number of dimensions
    p : time lag parameter, also called order of MAR model
    return values
    mar["A"] : model parameters is a NxNxP tensor, i.e. one NxN parameter matrix for each time bin k ∈ {1,...,p}
    mar["Σ"] : noise covariance matrix
"""
function mar_ml(y, p)
    (ns, nd) = size(y)
    ns < nd && error("error: there are more covariates than observation")
    y = transpose(y)
    Y = y[:, p+1:ns]
    X = zeros(nd*p, ns-p)
    for i = p:-1:1
        X[(p-i)*nd+1:(p-i+1)*nd, :] = y[:, i:ns+i-p-1]
    end

    A = (Y*X')/(X*X')
    ϵ = Y - A*X
    Σ = ϵ*ϵ'/ns     #(ns-p-p*nd-1)
    A = -[A[:, (i-1)*nd+1:i*nd] for i = 1:p]    # flip sign to be consistent with SPM12 convention
    mar = Dict([("A", A), ("Σ", Σ), ("p", p)])
    return mar
end


"""
Selection of model order based on MDL (this is also the negative BIC)
"""
function mdl4mar(Σ, ns, d, p)
    np = d^2 * p
    σ² = 1/ns * det((ns-np)*Σ)^(1/d) 
    mdl = (ns*σ² + np*log(ns))/2
    return mdl
end


"""
This function converts multivariate auto-regression (MAR) model parameters to a cross-spectral density (CSD).
A     : coefficients of MAR model, array of length p, each element contains the regression coefficients for that particular time-lag.
Σ     : noise covariance matrix of MAR
p     : number of time lags
freqs : frequencies at which to evaluate the CSD
sf    : sampling frequency

This function returns:
csd   : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
"""
function mar2csd(mar, freqs, sf)
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2*pi*freqs/sf
    nf = length(w)
	csd = zeros(ComplexF64, nf, nd, nd)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'
	end
    csd = 2*csd/sf
    return csd
end

function mar2csd(mar, freqs)
    sf = 2*freqs[end]   # freqs[end] is not the sampling frequency of the signal... not sure about this step.
    return mar2csd(mar, freqs, sf)
end


"""
Plain implementation of idft because AD dispatch versions for ifft don't work still!
"""
function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{eltype(x)}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end

"""
This function converts a cross-spectral density (CSD) into a multivariate auto-regression (MAR) model. It first transforms the CSD into its
cross-correlation function (Wiener-Kinchine theorem) and then computes the MAR model coefficients.
csd       : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
w         : frequencies
dt        : time step size
p         : number of time steps of auto-regressive model

This function returns
coeff     : array of length p of coefficient matrices of size sqrt(N)xsqrt(N)
noise_cov : noise covariance matrix
"""
function csd2mar(csd, w, dt, p)
    # TODO: investiagate why SymmetricToeplitz(ccf[1:p, i, j]) is not good to be used but instead need to use Toeplitz(ccf[1:p, i, j], ccf[1:p, j, i])
    # as is done in the original MATLAB code (confront comment there). ccf should be a symmetric matrix so there should be no difference between the
    # Toeplitz matrices but for the second jacobian (J[2], see for loop for i = 1:nJ in function diff) the computation produces subtle differences between
    # the two versions.

    dw = w[2] - w[1]
    w = w/dw
    ns = dt^-1
    N = ceil(Int64, ns/2/dw)
    gj = findall(x -> x > 0 && x < (N + 1), w)
    gi = gj .+ (ceil(Int64, w[1]) - 1)    # TODO: figure out what's the purpose of this!
    g = zeros(eltype(csd), N)

    # transform to cross-correlation function
    ccf = zeros(eltype(csd), N*2+1, size(csd,2), size(csd,3))
    for i = 1:size(csd, 2)
        for j = 1:size(csd, 3)
            g[gi] = csd[gj,i,j]
            f = idft(g)
            f = idft(vcat([0.0im; g; conj(g[end:-1:1])]))
            ccf[:,i,j] = real.(fftshift(f))*N*dw
        end
    end

    # MAR coefficients
    N = size(ccf,1)
    m = size(ccf,2)
    n = (N - 1) ÷ 2
    p = min(p, n - 1)
    ccf = ccf[(1:n) .+ n,:,:]
    A = zeros(eltype(csd), m*p, m)
    B = zeros(eltype(csd), m*p, m*p)
    for i = 1:m
        for j = 1:m
            A[((i-1)*p+1):i*p, j] = ccf[(1:p) .+ 1, i, j]
            B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = Toeplitz(ccf[1:p, i, j], vcat(ccf[1,i,j], ccf[2:p, j, i]))  # SymmetricToeplitz(ccf[1:p, i, j])
        end
    end
    a = B\A

    Σ  = ccf[1,:,:] - A'*a   # noise covariance matrix
    lags = [-a[i:p:m*p, :] for i = 1:p]
    mar = Dict([("A", lags), ("Σ", Σ), ("p", p)])
    return mar
end