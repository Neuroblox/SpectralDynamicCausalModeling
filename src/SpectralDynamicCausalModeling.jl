


#= Define notational equivalences between SPM and this code:

# the following two precision matrices will not be updated by the code,
# they belong to the assumed prior distribution p
ipC = Πθ_pr   # precision matrix of prior of parameters p(θ)
ihC = Πλ_pr   # precision matrix of prior of hyperparameters p(λ)

Variational distribution parameters:
pE, Ep = μθ_pr, μθ_po   # prior and posterior expectation of parameters (q(θ))
pC, Cp = θΣ, Σθ   # prior and posterior covariance of parameters (q(θ))
hE, Eh = μλ_pr, μλ   # prior and posterior expectation of hyperparameters (q(λ))
hC, Ch = λΣ, Σλ   # prior and posterior covariance of hyperparameters (q(λ))

Σ, iΣ  # data covariance matrix (likelihood), and its inverse (precision of likelihood - use Π only for those precisions that don't change)
Q      # components of iΣ; definition: iΣ = sum(exp(λ)*Q)
=#
