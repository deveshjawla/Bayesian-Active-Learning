using Distributed
using Turing
num_chains = 3
addprocs(num_chains; exeflags=`--project`)

using EvidentialFlux
using Flux
using Plots
using Distributions
using Turing
using DelimitedFiles
PATH = @__DIR__
cd(PATH)

function gendata(n)
    x1 = randn(Float32, 2, n)
    x2 = randn(Float32, 2, n) .+ [2, 2]
    x3 = randn(Float32, 2, n) .+ [-2, 2]
    y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
    y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
    y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
    hcat(x1, x2, x3), permutedims(hcat(y1, y2, y3))
end

# Generate data
n = 200
X, y = gendata(n)
Y = argmax.(eachcol(y))

input_size = size(X)[1]
output_size = size(y)[1]

# # Define model
# m = Chain(
#     Dense(input_size => 8, relu),
#     Dense(8 => 8, relu),
#     DIR(8 => output_size)
# )
# _, feedforward = Flux.destructure(m)

l1, l2 = 8, 8
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * output_size + output_size

num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:16], 8, 2)
    b0 = θ[17:24]
    W1 = reshape(θ[25:88], 8, 8)
    b1 = θ[89:96]
    W2 = reshape(θ[97:120], 3, 8)
    b2 = θ[121:123]

    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
		Dense(W2, b2),
		softmax
        # DIR(W2, b2)
    )
    return model
end

prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1),
    sqrt(2 / (l1 + l2)) * ones(nl2),
    sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))

num_params = lastindex(prior_std)

@model function BNN(x, y, feedforward, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
	# θ ~ Product([Cauchy(0,1) for _ in 1:num_params])
    nn = feedforward(θ)
	preds = nn(x)
    # α̂ = nn(x)
    # preds = mapslices(x -> x ./ sum(x), α̂, dims=1)

	# _labels=sort(unique(y))
	# onehot_y = mapreduce(x->Flux.onehot(x, _labels), hcat, y)
	# a = onehot_y .+ (1 .- onehot_y) .* α̂

    for i = 1:lastindex(y)
        y[i] ~ Categorical(preds[:, i])
    end
	# for i = 1:lastindex(y)
	# 	loglik = loglikelihood(Categorical(preds[:, i]), y[i]) + EvidentialFlux.kl(a[:,i])[1]
	# 	Turing.@addlogprob!(loglik)
	# end
end

using ReverseDiff
Turing.setadbackend(:reversediff)

N = 100
ch1_timed = @timed sample(BNN(X, Y, feedforward, num_params, prior_std), NUTS(), N)
ch1 = ch1_timed.value
elapsed = Float32(ch1_timed.time)

weights = MCMCChains.group(ch1, :θ).value #get posterior MCMC samples for network weights
params_set = collect.(Float32, eachrow(weights[:, :, 1]))
param_matrix = mapreduce(permutedims, vcat, params_set)

using StatsBase: countmap
function majority_voting(predictions::AbstractVector)::Vector{Float32}
    count_map = countmap(predictions)
    # println(count_map)
    uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
    index_max = argmax(nUniques)
    prediction = uniques[index_max]
    pred_probability = maximum(nUniques) / sum(nUniques)
    return [prediction, 1 - pred_probability] # 1 -  => give the unceratinty
end

uncertainty(α) = first(size(α)) ./ sum(α, dims=1)

# function pred_analyzer_multiclass(test_xs::Array{Float32,2}, params_set::Array{Float32,2})::Array{Float32,2}
#     nets = map(feedforward, eachrow(params_set))
#     predictions_nets = map(x -> x(test_xs), nets)
#     # predictions_nets = map(x -> x .+ 1, predictions_nets)
#     ŷ_prob = map(x -> mapslices(y -> y ./ sum(y), x, dims=1), predictions_nets) #ŷ
#     u = mapreduce(x -> mapreduce(uncertainty, hcat, eachcol(x)), vcat, predictions_nets)
#     ŷ_label = mapreduce(x -> mapreduce(argmax, hcat, eachcol(x)), vcat, ŷ_prob)
#     pred_plus_std = mapslices(majority_voting, ŷ_label, dims=1)
#     u_plus_std = mapslices(x -> [mean(x), std(x)], u, dims=1)
#     pred_matrix = vcat(pred_plus_std, u_plus_std)
#     return pred_matrix[[1, 3], :]
# end

function normalized_entropy(prob_vec::Vector, n_output)::Float32
	sum_probs = sum(prob_vec)
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum_probs < 0.99999 || sum_probs > 1.00001
        return error("sum(prob_vec) is not 1 BUT $(sum_probs) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log2.(prob_vec))) / log2(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::Matrix, n_output)
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    return H
end

"""
return H, E_H, H + E_H
"""
function bald(prob_matrix::Matrix, n_output)
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H - E_H
end

function pred_analyzer_multiclass(test_xs::Array{Float32, 2}, params_set::Array{Float32, 2})::Array{Float32, 2}
	nets = map(feedforward, eachrow(params_set))
	predictions_nets = map(x-> x(test_xs), nets)

	pred_matrix = cat(predictions_nets..., dims=3)
	bald_scores = mapslices(x -> bald(x, first(size(x))), pred_matrix, dims=[1, 3])
	aleatoric_uncertainties = mapreduce(x -> x[1], hcat, bald_scores[1, :, 1])
   
	ensembles = mapreduce(x-> mapslices(argmax, x, dims=1), vcat, predictions_nets)
	pred_plus_std = mapslices(majority_voting, ensembles, dims =1)
	pred_matrix = vcat(pred_plus_std, aleatoric_uncertainties)
    return pred_matrix[[1, 3], :]
end

# pred_analyzer_multiclass(X[:,403:405], param_matrix)

xs = Float32.(-5:0.1:5)
ys = Float32.(-5:0.1:5)
heatmap(xs, ys, (x, y) -> pred_analyzer_multiclass(reshape([x, y], (:, 1)), param_matrix)[2]) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./Layerwise_Gaussian_Elapsed_$(elapsed)_sec.pdf")
