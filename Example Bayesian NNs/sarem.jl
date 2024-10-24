using Flux, Distributions

struct Likelihood
    network
    sigma
end
Flux.@functor Likelihood #tell Flux to look for trainable parameters in Likelihood

(p::Likelihood)(x) = Normal.(p.network(x)[:], p.sigma[1]); #Flux only recognizes Matrix parameters but Normal() needs a scalar for sigma

likelihood = Likelihood(Chain(Dense(1, 5, relu), Dense(5, 1)), ones(1, 1))

params, likelihood_reconstructor = Flux.destructure(likelihood)
n_weights = lastindex(params) - 1

likelihood_conditional(weights, sigma) = likelihood_reconstructor(vcat(weights..., sigma))

weight_prior = MvNormal(zeros(n_weights), ones(n_weights))
# weight_prior = Product([Semicircle(0.5) for _ in 1:n_weights]);
sigma_prior = Gamma(1.0, 1.0);
Xline = Matrix{Float32}(transpose(collect(-3:0.1:3)[:, :]))
likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)

# , Plots
# Random.seed!(54321)
# plot(Xline[:],mean.(likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)),color=:red, legend=:none, fmt=:pdf)
# plot!(Xline[:],mean.(likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)),color=:red)
# plot!(Xline[:],mean.(likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)),color=:red)
# plot!(Xline[:],mean.(likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)),color=:red)
# plot!(Xline[:],mean.(likelihood_conditional(rand(weight_prior), rand(sigma_prior))(Xline)),color=:red)


#Training the Bayesian Neural Network

Random.seed!(54321)
X = rand(1, 50) .* 4 .- 2
y = sin.(X) .+ randn(1, 50) .* 0.25

scatter(X[:], y[:], color=:green, legend=:none, fmt=:pdf)

using Turing

@model function TuringModel(likelihood_conditional, weight_prior, sigma_prior, X, y)
    weights ~ weight_prior
    sigma ~ sigma_prior

    predictions = likelihood_conditional(weights, sigma)(X)

    y[:] ~ Product(predictions)
end



Random.seed!(54321)
N = 5000
ch = sample(TuringModel(likelihood_conditional, weight_prior, sigma_prior, X, y), NUTS(), N);


weights = Array(MCMCChains.group(ch, :weights).value) #get posterior MCMC samples for network weights


sigmas = Array(MCMCChains.group(ch, :sigma).value); #get posterior MCMC samples for standard deviation

Random.seed!(54321)

posterior_predictive_mean_samples = []
posterior_predictive_full_samples = []

for _ in 1:10000
    samp = rand(1:5000, 1)
    W = weights[samp, :, 1]
    sigma = sigmas[samp, :, 1]
    posterior_predictive_model = likelihood_reconstructor(vcat(W[:], sigma[:]))

    predictive_distribution = posterior_predictive_model(Xline)
    postpred_full_sample = rand(Product(predictive_distribution))
    push!(posterior_predictive_mean_samples, mean.(predictive_distribution))
    push!(posterior_predictive_full_samples, postpred_full_sample)
end

posterior_predictive_mean_samples = hcat(posterior_predictive_mean_samples...)

pp_mean = mean(posterior_predictive_mean_samples, dims=2)[:]
pp_mean_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_mean_samples, dims=2)[:]
pp_mean_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_mean_samples, dims=2)[:]

posterior_predictive_full_samples = hcat(posterior_predictive_full_samples...)
pp_full_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_full_samples, dims=2)[:]
pp_full_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_full_samples, dims=2)[:]

plot(Xline[:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), legend=:bottomright, label="Full posterior predictive distribution", fmt=:png)
plot!(Xline[:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (a.k.a. epistemic uncertainty)")
scatter!(X[:], y[:], color=:green, label="Training data")