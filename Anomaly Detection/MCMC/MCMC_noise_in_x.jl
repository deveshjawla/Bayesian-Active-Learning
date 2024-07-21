using EvidentialFlux
using Flux
using Plots
using Distributions
using Turing
using DelimitedFiles
using Statistics
PATH = @__DIR__
cd(PATH)

# Generate data
n = 20
X, y = gen_3_clusters(n)
Y = argmax.(eachcol(y))
# Y = y


test_X, test_y = gen_3_clusters(100)
test_Y = argmax.(eachcol(test_y))

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
n_output_layer = l2 * output_size

num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:16], 8, 2)
    b0 = θ[17:24]
    W1 = reshape(θ[25:88], 8, 8)
    b1 = θ[89:96]
    W2 = reshape(θ[97:120], 3, 8)

    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        DIR(W2, false),
        # DIR(W2, b2)
    )
    return model
end

prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1),
    sqrt(2 / (l1 + l2)) * ones(nl2),
    sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))

num_params = lastindex(prior_std)

@model function bnnnn(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    noise ~ MvNormal(zeros(2), ones(2))
    preds = nn(x .+ noise)
    for i = 1:lastindex(y)
        y[i] ~ Categorical((preds[:, i]) ./ sum(preds[:, i]))
    end
end

using ReverseDiff
Turing.setadbackend(:reversediff)

N = 1000
ch1_timed = @timed sample(bnnnn(X, Y, num_params, prior_std), NUTS(), N)
ch1 = ch1_timed.value
elapsed = Float32(ch1_timed.time)

weights = MCMCChains.group(ch1, :θ).value #get posterior MCMC samples for network weights
noises = MCMCChains.group(ch1, :noise).value #get posterior MCMC samples for network weights
params_set = collect.(Float32, eachrow(weights[:, :, 1]))
param_matrix = mapreduce(permutedims, vcat, params_set)

noise_set = collect.(Float32, eachrow(noises[:, :, 1]))
noise_matrix = mapreduce(permutedims, vcat, noise_set)

ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set)
ŷ = ŷ_uncertainties[1,:]
@info "Accuracy is" mean(test_Y.==ŷ)

test = pred_analyzer_multiclass(reshape([0.0,0.0], (:, 1)), param_matrix, noise_set)

xs = Float32.(-10:0.5:10)
ys = Float32.(-10:0.5:10)
heatmap(xs, ys, (x, y) -> pred_analyzer_multiclass(reshape([x, y], (:, 1)), param_matrix, noise_set)[3]) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./noise_in_x_$(elapsed)_sec.pdf")
