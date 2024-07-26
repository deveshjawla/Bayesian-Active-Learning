using EvidentialFlux
using Flux
using Plots
PATH = @__DIR__
cd(PATH)

# Generate data

n = 200
X, y = gen_3_clusters(n)

input_size = size(X, 1)
output_size = size(y, 1)

l1, l2 = 8, 8
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * output_size #+ output_size

num_params = nl1 + nl2 + n_output_layer

# train_y = Flux.onehotbatch(vec(train_y), 1:output_size)
# println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
# # println(eltype(train_x), eltype(train_y))
train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)
if !isnothing(sample_weights)
    sample_weights_loader = Flux.DataLoader(permutedims(sample_weights), batchsize=acq_size)
else
    sample_weights_loader = nothing
end

using Statistics

include("./../AdaBeliefCosAnnealNNTraining.jl")
optim_theta, re = network_training(m, input_size, output_size, n_epochs, train_loader, sample_weights_loader)
# pred_analyzer_multiclass(re, X, optim_theta)

# Test predictions
α̂ = m(X)
ŷ = α̂ ./ sum(α̂, dims=1)
u = edlc_uncertainty(α̂)

xs = Float32.(-10:0.1:10)
ys = Float32.(-10:0.1:10)
# Show epistemic uncertainty
heatmap(xs, ys, (x, y) -> edlc_uncertainty(m(vcat(x, y)))[1])
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./relu_edlc.pdf")