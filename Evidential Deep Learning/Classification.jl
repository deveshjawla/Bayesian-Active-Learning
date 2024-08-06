using Flux
using StatsPlots
using DataFrames
PATH = @__DIR__
cd(PATH)
include("./DirichletLayer.jl")
# Generate data
include("./../DataUtils.jl")
include("./../AdaBeliefCosAnnealNNTraining.jl")

# train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)
# if !isnothing(sample_weights)
#     sample_weights_loader = Flux.DataLoader(permutedims(sample_weights), batchsize=acq_size)
# else
#     sample_weights_loader = nothing
# end

# Generate data
n = 20
train_x, train_y = gen_3_clusters(n)

n_input = size(train_x, 1)
n_output = size(train_y, 1)

optim_theta, re = network_training("Evidential Classification", n_input, n_output, 100; data=(train_x, train_y), loss_function=dirloss)
m=re(optim_theta)
# Test predictions
α̂ = m(train_x)
ŷ = α̂ ./ sum(α̂, dims=1)
u = edlc_uncertainty(α̂)

X1 = Float32.(-4:0.1:4)
X2 = Float32.(-3:0.1:5)
# Show epistemic uncertainty
test_x_area = pairs_to_matrix(X1, X2)
ŷ_uncertainty = edlc_uncertainty(m(test_x_area))
uncertainties = reshape(ŷ_uncertainty, (lastindex(X1), lastindex(X2)))
gr(size=(700, 600), dpi=300)
heatmap(X1, X2, uncertainties)
scatter!(train_x[1, train_y[1, :].==1], train_x[2, train_y[1, :].==1], color=:red, label="1")
scatter!(train_x[1, train_y[2, :].==1], train_x[2, train_y[2, :].==1], color=:green, label="2")
scatter!(train_x[1, train_y[3, :].==1], train_x[2, train_y[3, :].==1], color=:blue, label="3", legend_title="Classes", aspect_ratio=:equal, xlim=[-4, 4], ylim=[-3, 5], colorbar_title="Evidential Deep Learning Uncertainty")
savefig("./EDLC.pdf")