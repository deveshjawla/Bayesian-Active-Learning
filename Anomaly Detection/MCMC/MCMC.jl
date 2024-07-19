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
X, y = gendata(n)
Y = argmax.(eachcol(y))
# Y = y


test_X, test_y = gendata(100)
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
        Dense(W2, false),
		softmax
        # DIR(W2, b2)
    )
    return model
end

prior_std = Float64.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1),
    sqrt(2 / (l1 + l2)) * ones(nl2),
    sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))

num_params = lastindex(prior_std)

@model function bnnnn(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    preds = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(preds[:, i])
    end
end

using ReverseDiff
Turing.setadbackend(:reversediff)

N = 1000
ch1_timed = @timed sample(bnnnn(X, Y, num_params, prior_std), NUTS(), N)
ch1 = ch1_timed.value
elapsed = Float64(ch1_timed.time)

weights = MCMCChains.group(ch1, :θ).value #get posterior MCMC samples for network weights
params_set = collect.(Float64, eachrow(weights[:, :, 1]))
param_matrix = mapreduce(permutedims, vcat, params_set)


uncertainty(α) = first(size(α)) ./ sum(α, dims=1)







ŷ = pred_analyzer_multiclass(test_X, param_matrix)[1,:]
@info "Accuracy is" mean(test_Y.==ŷ)

xs = Float64.(-5:0.1:5)
ys = Float64.(-5:0.1:5)
heatmap(xs, ys, (x, y) -> pred_analyzer_multiclass(reshape([x, y], (:, 1)), param_matrix)[4]) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./no_noise_$(elapsed)_sec.pdf")
