using Flux
using StatsPlots
using DataFrames
PATH = @__DIR__
cd(PATH)
# Generate data
include("./../DataUtils.jl")

# Generate data
n = 20
train_X, train_y = gen_3_clusters(n)
train_Y = mapslices(argmax, train_y, dims=1)

test_X, test_y = gen_3_clusters(100)
test_Y = mapslices(argmax, test_y, dims=1)

input_size = size(train_X, 1)
output_size = size(train_y, 1)

l1, l2 = 8, 8
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * output_size

num_params = nl1 + nl2 + n_output_layer

include("./../MCMC Variants Comparison/SoftmaxNetwork3.jl")
prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))
location = zeros(num_params)

using Turing, Flux, ReverseDiff
include("./../BayesianModel.jl")

model = BNN(train_X, train_Y, location, prior_std)

using Flux, Turing
using Turing: Variational
q0 = Variational.meanfield(model)
advi = ADVI(10, 1000; adtype=AutoReverseDiff())
opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(model, advi, q0; optimizer=opt)
z = rand(q, 1000)

using Bijectors
_, sym2range = bijector(model, Val(true))

params_set = Float32.(z[first(sym2range.θ), :])
param_matrix = permutedims(params_set)
noise_set_x = collect(eachcol(Float32.(z[first(sym2range.noise_x), :])))

using Distributed
using DelimitedFiles
using CategoricalArrays
include("./../MCMCUtils.jl")
include("./../ScoringFunctions.jl")
ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix; noise_set_x=noise_set_x)
ŷ = ŷ_uncertainties[1, :]
acc, f1 = performance_stats_multiclass(vec(Float32.(test_Y)), ŷ)
@info "Balanced Accuracy and F1 are " acc f1

using StatsPlots
X1 = Float32.(-4:0.1:4)
X2 = Float32.(-3:0.1:5)

test_x_area = pairs_to_matrix(X1, X2)

ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set_x=noise_set_x)
uncertainty_metrics = ["Predicted Label", "Prediction Probability", "Std of Prediction Probability", "Aleatoric Uncertainty", "Epistemic Uncertainty", "Total Uncertainty"]

for (i, j) in enumerate(uncertainty_metrics)

    uncertainties = reshape(ŷ_uncertainties[i, :], (lastindex(X1), lastindex(X2)))

    gr(size=(700, 600), dpi=300)
    heatmap(X1, X2, uncertainties)
    scatter!(train_X[1, train_y[1, :].==1], train_X[2, train_y[1, :].==1], color=:red, label="1")
    scatter!(train_X[1, train_y[2, :].==1], train_X[2, train_y[2, :].==1], color=:green, label="2")
    scatter!(train_X[1, train_y[3, :].==1], train_X[2, train_y[3, :].==1], color=:blue, label="3", legend_title="Classes", title="Variational Inference", aspect_ratio=:equal, xlim=[-4, 4], ylim=[-3, 5], colorbar_title=" \n$(j)")

    savefig("./$(j).pdf")
end

# q.dist
# q.dist.m
# q.dist.σ
# cov(q.dist)

# elbo(advi, q, model, 1000)

# histogram(z[1, :])
# avg[union(sym2range[:noise_x]...)]

# function plot_variational_marginals(z, sym2range)
#     ps = []

#     for (i, sym) in enumerate(keys(sym2range))
#         indices = union(sym2range[sym]...)  # <= array of ranges
#         if sum(length.(indices)) > 1
#             offset = 1
#             for r in indices
#                 p = density(
#                     z[r, :];
#                     title="$(sym)[$offset]",
#                     titlefontsize=10,
#                     label="",
#                     ylabel="Density"
#                 )
#                 push!(ps, p)
#                 offset += 1
#             end
#         else
#             p = density(
#                 z[first(indices), :];
#                 title="$(sym)",
#                 titlefontsize=10,
#                 label="",
#                 ylabel="Density"
#             )
#             push!(ps, p)
#         end
#     end

#     return plot(ps...; layout=(length(ps), 1), size=(500, 2000))
# end

# plot_variational_marginals(z, sym2range.noise_x)