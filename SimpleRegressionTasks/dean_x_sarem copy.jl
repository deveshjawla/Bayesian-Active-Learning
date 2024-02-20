using Flux
using Plots
using Distributions
using Turing
using DelimitedFiles
PATH = @__DIR__
cd(PATH)

experiment = "synthetic_data"

using Random
# f(x,z) = 3*x + 4*z^2
x_test = collect(Float32, 4:0.05:4.95) .+ rand(MersenneTwister(0), Float32, 20)
z_test = collect(Float32, 1:0.03:1.57) .+ rand(MersenneTwister(0), Float32, 20)
x_train = collect(Float32, 3:0.05:3.95) .+ rand(MersenneTwister(1), Float32, 20)
z_train = collect(Float32, 0:0.03:0.57) .+ rand(MersenneTwister(1), Float32, 20)
y(x, z) = 3 * x + 4 * z^2
y_train = [y(x, z) for (x, z) in zip(x_train, z_train)]
y_test = [y(x, z) for (x, z) in zip(x_test, z_test)]

train=permutedims(hcat(x_train, z_train))
test=permutedims(hcat(x_test, z_test))

# n_output=1
# l1, l2 = 5, 5
# nl1 = 2 * l1 + l1
# nl2 = l1 * l2 + l2
# n_output_layer = l2 * n_output + n_output

# total_num_params = nl1 + nl2 + n_output_layer

# function nn_forward(θ::AbstractVector)
#     W0 = reshape(θ[1:10], 5, 2)
#     b0 = θ[11:15]
#     W1 = reshape(θ[16:40], 5, 5)
#     b1 = θ[41:45]
#     W2 = reshape(θ[46:50], 1, 5)
#     b2 = θ[51:51]

#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2)
#     )
#     return model
# end
# num_params = 51


function unpack(nn_params::AbstractVector)
	w11 = reshape(nn_params[1:4], 2, 2)
	b11 = reshape(nn_params[5:6], 2)
	w12 = reshape(nn_params[7:10], 2, 2)
	b12 = reshape(nn_params[11:12], 2)
	w13 = reshape(nn_params[13:16], 2, 2)
	b13 = reshape(nn_params[17:18], 2)

	w21 = reshape(nn_params[19:30], 2, 6)
	b21 = reshape(nn_params[31:32], 2)
	w22 = reshape(nn_params[33:44], 2, 6)
	b22 = reshape(nn_params[45:46], 2)
	w23 = reshape(nn_params[47:58], 2, 6)
	b23 = reshape(nn_params[59:60], 2)

	w31 = reshape(nn_params[61:66], 1, 6)
	b31 = reshape(nn_params[67:67], 1)
	# w32 = reshape(nn_params[29], 1, 1)
	# b32 = reshape(nn_params[30], 1)
	# w33 = reshape(nn_params[31], 1, 1)
	# b33 = reshape(nn_params[32], 1)
	return w11, w12, w13, b11, b12, b13, w21, w22, w23, b21, b22, b23, w31, b31#, b32, b33, w32, w33
end
num_params = 67

function nn_forward(nn_params::AbstractVector)
	w11, w12, w13, b11, b12, b13, w21, w22, w23, b21, b22, b23, w31, b31 = unpack(nn_params)
	nn = Chain(
		Parallel(vcat, Dense(w11, b11,sin), Dense(w12, b12, relu), Dense(w13, b13, identity)),
		Parallel(vcat, Dense(w21, b21, sin), Dense(w22, b22, relu), Dense(w23, b23, identity)), 
		Dense(w31, b31))
	return nn
end

@model bayes_nn(xs, ys, num_params) = begin
    nn_params ~ MvNormal(zeros(Float32, num_params), ones(Float32, num_params)) #Prior
    nn = nn_forward(nn_params)
    preds = nn(xs) #Build the net
    # preds = feedforward(xs, nn_params)
    sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
    for i = 1:lastindex(ys)
        ys[i] ~ Normal(preds[i], sigma)
    end
end

N = 1000
ch1_timed = @timed sample(bayes_nn(train, y_train, num_params), NUTS(), N)
ch1 = ch1_timed.value
elapsed = Float32(ch1_timed.time)
# ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);

weights = Array(MCMCChains.group(ch1, :nn_params).value) #get posterior MCMC samples for network weights
println(size(weights))
sigmas = Array(MCMCChains.group(ch1, :sigma).value)

mkpath("./$(experiment)/Plots")

posterior_predictive_mean_samples = []
posterior_predictive_full_samples = []

for _ in 1:10000
    # samp = rand(round(Int, 0.8 * N):N, 1)
    samp = rand(1:10:N, 1)
    W = weights[samp, :, 1]
    # W = weights[samp, :]
    sigma = sigmas[samp, :, 1]
    # sigma = sigmas[samp, :]
    nn = nn_forward(vec(W))
    # nn = destructured(vec(W))
    predictive_distribution = vec(Normal.(nn(test), sigma[1]))
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

plt = scatter(test[1,:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), legend=:bottom, label="Full posterior predictive distribution", fmt=:pdf, size=(600, 400), dpi=600)
scatter!(test[1,:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (epistemic uncertainty)")


# lp, maxInd = findmax(ch1[:lp])

# params, internals = ch1.name_map
# bestParams = map(x -> ch1[x].data[maxInd], params[1:num_params])
# plot!(test, vec(nn_forward(bestParams)(permutedims(test))), seriestype=:line, label="MAP Estimate", color=:navy)
# plot!(test, vec(destructured(bestParams)(permutedims(test))), seriestype=:line, label="MAP Estimate", color=:navy)
scatter!(train[1,:], y_train, color=:green, markersize=2.0, label="Training data", markerstrokecolor=:green)
scatter!(test[1,:], y_test, label="Truth", color=:green)

savefig(plt, "./$(experiment)/Plots/$(function_name)_$(activation_function).pdf")

# lPlot = plot(ch1[:lp], label="Chain 1", title="cos Posterior")
# # plot!(lPlot, ch2[:lp], label="Chain 2")

# sigPlot = plot(ch1[:sigma], label="Chain 1", title="Variance")
# # plot!(sigPlot, ch2[:sigma], label="Chain 2")

# plot(lPlot, sigPlot)
