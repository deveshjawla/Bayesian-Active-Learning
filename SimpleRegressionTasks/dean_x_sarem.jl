using Flux
using Plots
using Distributions
using Turing
using DelimitedFiles
PATH = @__DIR__
cd(PATH)
# input_size = 1
# firstlayer = Parallel(
#     vcat, 
#     Dense(input_size => 3, relu), 
#     Dense(input_size => 1, relu), 
#     Dense(input_size => 1, relu),  
# )
# outdims = size(firstlayer(rand(1)))

# secondlayer = Parallel(
#     vcat,
#     Dense(5 => 1, relu), 
#     Dense(5 => 1, relu), 
#     Dense(5 => 1, relu),  
#     # Dense(33 => 3, relu),  
#     # Dense(33 => 3, cos),  
#     # Dense(33 => 3, x->x^2),
# 	# Dense(33 => 3, x->x^3),
# 	# Dense(33 => 3, x->x^4),
# 	# Dense(33 => 3, x->x^5),
# 	# Dense(33 => 3, x->x^6),
# 	# Dense(input_size => 5, x->x^(1/2)),
# 	# Dense(input_size => 5, x->x^(1/3)),
# 	# Dense(input_size => 5, x->x^(1/4)),
# 	# Dense(input_size => 5, x->x^(1/5)),
# 	# Dense(input_size => 5, x->x^(1/6)),    
# )
# outdims = size(secondlayer(rand(12)))


# output_size = 1
# # skiplayer = SkipConnection(secondlayer, vcat)
# # outerlayer = Dense(24 => output_size) 
# # model = Chain(firstlayer, skiplayer, outerlayer)
# outerlayer = Dense(12 => output_size) 
# # model = Chain(firstlayer, secondlayer, outerlayer)

# model = Chain(Dense(1, 12, relu), Dense(12, 12, relu), Dense(12, 1))

# parameters_initial, destructured = Flux.destructure(model)
# feedforward(x, theta) = destructured(theta)(x)
# num_params = length(parameters_initial)

# il=Dense(2, 3)
# h1=Dense(3, 3)
# ol=Dense(6, 1)
# skip=SkipConnection(h1, vcat)
# model = Chain(il, skip, ol)
activation_function = "mix"
temperatures = [1.0, 0.1, 0.001]
function_names = ["Cosine", "Polynomial", "Exponential", "Logarithmic", "sin(pisin)"]#  "Cosine", "Polynomial", "Exponential", "Logarithmic",

function f(x, name::String)
    if name == "Polynomial"
        f = x^4 + 2 * x^3 - 3 * x^2 + x
    elseif name == "Cosine"
        f = cos(x)#sin(π * sin(x))
    elseif name == "sin(pisin)"
        f = sin(π * sin(x))
    elseif name == "Exponential"
        f = exp(x)
    elseif name == "Logarithmic"
        f = log(x)
    end
    return f
end


for function_name in function_names
	for temperature in temperatures
	experiment = "custom_loglikelihood_MCMC_$(temperature)"

    begin
        if function_name == "Polynomial"
            # f(x) = x^4 + 2 * x^3 - 3 * x^2 + x 
            xs1 = collect(Float32, -3.36:0.02:-2.9)# .
            xs2 = collect(Float32, -1:0.02:1.8)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -3.6:0.01:2.3)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)
        elseif function_name == "Cosine" || function_name == "sin(pisin)"
            # f(x) = cos(x) 
            xs1 = collect(Float32, -7:0.05:-1)# .
            xs2 = collect(Float32, 3:0.05:5)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -15:0.01:15)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)
        elseif function_name == "Exponential"
            # f(x) = exp(x) 
            xs1 = collect(Float32, -1:0.01:1)# .
            xs2 = collect(Float32, 2:0.01:4.5)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -2:0.01:5)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)

        elseif function_name == "Logarithmic"
            # f(x) = log(x) 
            xs1 = collect(Float32, 0.01:0.005:1)# .
            xs2 = collect(Float32, 2:0.01:3)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, 0:0.005:6)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)

        end
    end

    function unpack(nn_params::AbstractVector)
        w11 = reshape(nn_params[1:3], 3, 1)
        b11 = reshape(nn_params[4:6], 3)
        w12 = reshape(nn_params[7:7], 1, 1)
        b12 = reshape(nn_params[8:8], 1)
        w13 = reshape(nn_params[9:9], 1, 1)
        b13 = reshape(nn_params[10:10], 1)

        w21 = reshape(nn_params[11:15], 1, 5)
        b21 = reshape(nn_params[16:16], 1)
        w22 = reshape(nn_params[17:21], 1, 5)
        b22 = reshape(nn_params[22:22], 1)
        w23 = reshape(nn_params[23:27], 1, 5)
        b23 = reshape(nn_params[28:28], 1)

        w31 = reshape(nn_params[29:31], 1, 3)
        b31 = reshape(nn_params[32:32], 1)
        # w32 = reshape(nn_params[29], 1, 1)
        # b32 = reshape(nn_params[30], 1)
        # w33 = reshape(nn_params[31], 1, 1)
        # b33 = reshape(nn_params[32], 1)
        return w11, w12, w13, b11, b12, b13, w21, w22, w23, b21, b22, b23, w31, b31#, b32, b33, w32, w33
    end

    function nn_forward(nn_params::AbstractVector)
        w11, w12, w13, b11, b12, b13, w21, w22, w23, b21, b22, b23, w31, b31 = unpack(nn_params)
        nn = Chain(
            Parallel(vcat, Dense(w11, b11,relu), Dense(w12, b12, relu), Dense(w13, b13, relu)),
			Parallel(vcat, Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23,relu)), 
			Dense(w31, b31))
        return nn
    end

    # function unpack(nn_params::AbstractVector)
    #     W₁ = reshape(nn_params[1:5], 5, 1)
    #     b₁ = reshape(nn_params[6:10], 5)
    #     W₂ = reshape(nn_params[11:25], 3, 5)
    #     b₂ = reshape(nn_params[26:28], 3)
    #     W₃ = reshape(nn_params[29:31], 1, 3)
    #     b₃ = reshape(nn_params[32:32], 1)
    #     return W₁, b₁, W₂, b₂, W₃, b₃
    # end
	# function nn_forward(nn_params::AbstractVector)
    #     W₁, b₁, W₂, b₂, W₃, b₃ = unpack(nn_params)
    #     nn = Chain(Dense(W₁, b₁, relu), Dense(W₂, b₂, relu), Dense(W₃, b₃))
    #     return nn
    # end

	# function unpack(nn_params::AbstractVector)
    #     W₁ = reshape(nn_params[1:5], 5, 1)
    #     b₁ = reshape(nn_params[6:10], 5)
    #     W₂ = reshape(nn_params[11:25], 3, 5)
    #     b₂ = reshape(nn_params[26:28], 3)
	# 	W3 = reshape(nn_params[29:37], 3, 3)
    #     b3 = reshape(nn_params[38:40], 3)
	# 	W4 = reshape(nn_params[40:48], 3, 3)
    #     b4 = reshape(nn_params[49:51], 3)
    #     W5 = reshape(nn_params[52:54], 1, 3)
    #     b5 = reshape(nn_params[55:55], 1)
    #     return W₁, b₁, W₂, b₂, W3, b3, W4, b4, W5, b5
    # end
    # function nn_forward(nn_params::AbstractVector)
    #     W₁, b₁, W₂, b₂, W3, b3, W4, b4, W5, b5 = unpack(nn_params)
    #     nn = Chain(Dense(W₁, b₁, relu), Dense(W₂, b₂, relu), Dense(W3, b3, relu), Dense(W4, b4, relu), Dense(W5, b5))
    #     return nn
    # end

    glorot_normal = Float32.(vcat(sqrt(2 / (1 + 5)) * ones(10), sqrt(2 / (5 + 3)) * ones(18), sqrt(2 / (3 + 1)) * ones(4)))
    # glorot_normal = Float32.(vcat(sqrt(2 / (1 + 5)) * ones(10), sqrt(2 / (5 + 3)) * ones(18), sqrt(2 / (3 + 3)) * ones(23), sqrt(2 / (3 + 1)) * ones(4)))
    # glorot_normal = Float32.(vcat(sqrt(2 / (1 + 12)) * ones(24), sqrt(2 / (12 + 12)) * ones(156), sqrt(2 / (12 + 1)) * ones(13)))
    num_params = lastindex(glorot_normal)

    @model bayes_nn(xs, ys, num_params, prior_variances, Temp) = begin
        nn_params ~ MvNormal(zeros(Float32, num_params), prior_variances) #Prior
        nn = nn_forward(nn_params)
        preds = nn(xs) #Build the net
        # preds = feedforward(xs, nn_params)
        sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
        for i = 1:lastindex(ys)
            # ys[i] ~ Normal(preds[i], sigma)
			loglik = loglikelihood(Normal(preds[i], sigma), ys[i])/Temp
			Turing.@addlogprob!(loglik)
        end
    end

	

    N = 1000
    ch1_timed = @timed sample(bayes_nn(hcat(xs...), ys, num_params, glorot_normal, temperature), NUTS(), N)
	ch1 = ch1_timed.value
	elapsed = Float32(ch1_timed.time)
    # ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);

    weights = Array(MCMCChains.group(ch1, :nn_params).value) #get posterior MCMC samples for network weights
	println(size(weights))
    sigmas = Array(MCMCChains.group(ch1, :sigma).value)

	mkpath("./$(experiment)/Plots")
	writedlm("$(experiment)/$(function_name)_$(activation_function)_weights.csv", weights[:,:,1], ',')
	writedlm("$(experiment)/$(function_name)_$(activation_function)_sigmas.csv", sigmas, ',')
	writedlm("$(experiment)/$(function_name)_$(activation_function)_elapsed.csv", elapsed, ',')

    # weights= readdlm("$(experiment)/$(function_name)_$(activation_function)_weights.csv", ',')
	# sigmas=readdlm("$(experiment)/$(function_name)_$(activation_function)_sigmas.csv", ',')

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

    plt = plot(Xline[:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), legend=:none, label="Full posterior predictive distribution", fmt=:pdf, size=(600, 400), dpi=600)
	
    plot!(Xline[:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (epistemic uncertainty)")


    # lp, maxInd = findmax(ch1[:lp])

    # params, internals = ch1.name_map
    # bestParams = map(x -> ch1[x].data[maxInd], params[1:num_params])
    # plot!(Xline, vec(nn_forward(bestParams)(permutedims(Xline))), seriestype=:line, label="MAP Estimate", color=:navy)
    # plot!(Xline, vec(destructured(bestParams)(permutedims(Xline))), seriestype=:line, label="MAP Estimate", color=:navy)
    scatter!(xs, ys, color=:green, markersize=2.0, label="Training data", markerstrokecolor=:green)
    plot!(Xline, map(x -> f(x, function_name), Xline), label="Truth", color=:green)

    savefig(plt, "./$(experiment)/Plots/$(function_name)_$(activation_function).pdf")
end
end

# lPlot = plot(ch1[:lp], label="Chain 1", title="cos Posterior")
# # plot!(lPlot, ch2[:lp], label="Chain 2")

# sigPlot = plot(ch1[:sigma], label="Chain 1", title="Variance")
# # plot!(sigPlot, ch2[:sigma], label="Chain 2")

# plot(lPlot, sigPlot)
