# Import libraries.
using Turing, Flux, Random
using LazyArrays, DistributionsAD


# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2*M); zeros(2*M)]
xs = hcat(xs...)

# Specify the network architecture.
network_shape = [
    (3,2, :relu),
    (2,3, :relu), 
    (1,2, :σ)]

# Regularization, parameter variance, and total number of
# parameters.
alpha = 0.09
sig = sqrt(1.0 / alpha)
num_params = sum([i * o + i for (i, o, _) in network_shape])

# This modification of the unpack function generates a series of vectors
# given a network shape.
# function unpack(θ::AbstractVector, network_shape::AbstractVector)
#     index = 1
#     weights = []
#     biases = []
#     for layer in network_shape
#         rows, cols, _ = layer
#         size = rows * cols
#         last_index_w = size + index - 1
#         last_index_b = last_index_w + rows
#         push!(weights, reshape(θ[index:last_index_w], rows, cols))
#         push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
#         index = last_index_b + 1
#     end
#     return deepcopy(weights), deepcopy(biases)
# end

# # Generate an abstract neural network given a shape, 
# # and return a prediction.
# function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
#     weights, biases = unpack(θ, network_shape)
#     layers = []
#     for i in eachindex(network_shape)
#         push!(layers, Dense(weights[i],
#             biases[i],
#             eval(network_shape[i][3])))
#     end
#     nn = Chain(layers...)
#     return nn(x)
# end

nn_initial = Chain(Dense(2, 3, relu), Dense(3, 2, relu), Dense(2, 1, sigmoid))

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, destructured = Flux.destructure(nn_initial)

feedforward(x, theta) = destructured(theta)(x)

# General Turing specification for a BNN model.
@model bayes_nn_general(xs, y, network_shape, num_params) = begin
    # θ ~ MvNormal(zeros(num_params), ones(num_params))
    θ ~ filldist(Normal(), num_params)
    # preds = nn_forward(xs, θ, network_shape)
    preds = feedforward(xs, θ)
	# preds = deepcopy(vec(permutedims(preds)))
    # for i = 1:lastindex(y)
	# 	y[i] ~ Bernoulli(preds[i])
	# end
	y ~ Product(Bernoulli.(preds))
	# y ~ arraydist(BroadcastArray(Bernoulli, preds))
end

# using ReverseDiff
# # Set the backend.
# Turing.setadbackend(:reversediff)
using Zygote
# Set the backend.
Turing.setadbackend(:zygote)

# Perform inference.
num_samples = 50
chain_timed = @timed sample(bayes_nn_general(xs, ts, network_shape, num_params), NUTS(), num_samples, progress=true)

chain_timed.time





using Bijectors
using Turing: Variational

m = bayes_nn(hcat(xs...), ts);

q = Variational.meanfield(m)

μ = randn(length(q))
ω = -1 .* ones(length(q))

q = Variational.update(q, μ, exp.(ω));

advi = ADVI(10, 1000)
q_hat = vi(m, advi, q);

samples = transpose(rand(q_hat, 5000))
ch_vi = Chains(reshape(samples, size(samples)..., 1), ["nn_params[$$i]" for i = 1:20]);

# Extract all weight and bias parameters.
theta = ch_vi[:nn_params].value.data;

# Plot the average prediction.
plot_data()

n_end = 1500
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_predict([x, y], theta, n_end)[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)