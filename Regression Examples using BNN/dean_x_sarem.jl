using Flux
using Plots
using Distributions
using Turing

activation_function = "tanh"
function_names = ["Polynomial", "Cosine", "Exponential", "Logarithmic"]# 

function f(x, name::String)
    if name == "Polynomial"
        f = x^4 + 2 * x^3 - 3 * x^2 + x
    elseif name == "Cosine"
        f = cos(x)
    elseif name == "Exponential"
        f = exp(x)
    elseif name == "Logarithmic"
        f = log(x)
    end
    return f
end


for function_name in function_names
    if function_name == "Polynomial"
        # f(x) = x^4 + 2 * x^3 - 3 * x^2 + x 
        xs1 = collect(-3.36:0.02:-2.9)# .
        xs2 = collect(-1:0.02:1.8)# .
        xs = vcat(xs1, xs2)
        Xline = collect(-3.6:0.01:2)
        ys = map(x -> f(x, function_name) + rand(Normal(0, 0.1)), xs)
    elseif function_name == "Cosine"
        # f(x) = cos(x) 
        xs1 = collect(-7:0.05:-1)# .
        xs2 = collect(3:0.05:5)# .
        xs = vcat(xs1, xs2)
        Xline = collect(-10:0.01:10)
        ys = map(x -> f(x, function_name) + rand(Normal(0, 0.1)), xs)
    elseif function_name == "Exponential"
        # f(x) = exp(x) 
        xs1 = collect(-1:0.01:1)# .
        xs2 = collect(2:0.01:4.5)# .
        xs = vcat(xs1, xs2)
        Xline = collect(-2:0.01:5)
        ys = map(x -> f(x, function_name) + rand(Normal(0, 0.1)), xs)

    elseif function_name == "Logarithmic"
        # f(x) = log(x) 
        xs1 = collect(0.01:0.005:1)# .
        xs2 = collect(2:0.01:3)# .
        xs = vcat(xs1, xs2)
        Xline = collect(0:0.005:6)
        ys = map(x -> f(x, function_name) + rand(Normal(0, 0.1)), xs)

    end


    function unpack(nn_params::AbstractVector)
        W₁ = reshape(nn_params[1:5], 5, 1)
        b₁ = reshape(nn_params[6:10], 5)
        W₂ = reshape(nn_params[11:25], 3, 5)
        b₂ = reshape(nn_params[26:28], 3)
        W₃ = reshape(nn_params[29:31], 1, 3)
        b₃ = reshape(nn_params[32:32], 1)
        return W₁, b₁, W₂, b₂, W₃, b₃
    end
    # if activation_function == "tanh"
    function nn_forward(nn_params::AbstractVector)
        W₁, b₁, W₂, b₂, W₃, b₃ = unpack(nn_params)
        nn = Chain(Dense(W₁, b₁, tanh), Dense(W₂, b₂, tanh), Dense(W₃, b₃))
        return nn
    end
    # elseif activation_function == "tanh"
    # 	function nn_forward(nn_params::AbstractVector)
    # 		W₁, b₁, W₂, b₂, W₃, b₃ = unpack(nn_params)
    # 		nn = Chain(Dense(W₁, b₁, tanh), Dense(W₂, b₂, tanh), Dense(W₃, b₃))
    # 		return nn
    # 	end
    # end

    init_variance = vcat(sqrt(2 / (1 + 5)) * ones(10), sqrt(2 / (5 + 3)) * ones(18), sqrt(2 / (3 + 1)) * ones(4))

    @model bayes_nn(xs, ys, prior_variances) = begin
        nn_params ~ MvNormal(zeros(32), prior_variances) #Prior
        nn = nn_forward(nn_params)
        preds = nn(xs) #Build the net
        sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
        for i = 1:lastindex(ys)
            ys[i] ~ Normal(preds[i], sigma)
        end
    end

    N = 1000
    ch1 = sample(bayes_nn(hcat(xs...), ys, init_variance), NUTS(), N)
    # ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);

    weights = Array(MCMCChains.group(ch1, :nn_params).value) #get posterior MCMC samples for network weights
    sigmas = Array(MCMCChains.group(ch1, :sigma).value)
    using Random
    Random.seed!(54321)

    posterior_predictive_mean_samples = []
    posterior_predictive_full_samples = []

    for _ in 1:10000
        samp = rand(round(Int, 0.8 * N):N, 1)
        W = weights[samp, :, 1]
        sigma = sigmas[samp, :, 1]
        nn = nn_forward(vec(W))
        predictive_distribution = vec(Normal.(nn(permutedims(Xline)), sigma[1]))
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

    plt = plot(Xline[:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), legend=:none, label="Full posterior predictive distribution", fmt=:png, size=(600, 400), dpi=600)
    plot!(Xline[:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (epistemic uncertainty)")


    lp, maxInd = findmax(ch1[:lp])

    params, internals = ch1.name_map
    bestParams = map(x -> ch1[x].data[maxInd], params[1:32])
    plot!(Xline, vec(nn_forward(bestParams)(permutedims(Xline))), seriestype=:line, label="MAP Estimate", color=:navy)
    scatter!(xs, ys, color=:green, label="Training data", markerstrokecolor=:green)
    plot!(Xline, map(x -> f(x, function_name), Xline), label="Truth", color=:black)

    savefig(plt, "Plots/$(function_name)_$(activation_function).png")

end

# lPlot = plot(ch1[:lp], label="Chain 1", title="cos Posterior")
# # plot!(lPlot, ch2[:lp], label="Chain 2")

# sigPlot = plot(ch1[:sigma], label="Chain 1", title="Variance")
# # plot!(sigPlot, ch2[:sigma], label="Chain 2")

# plot(lPlot, sigPlot)