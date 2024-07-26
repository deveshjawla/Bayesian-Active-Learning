using Flux #Deep learning
using Plots
using LaplaceRedux

PATH = @__DIR__
cd(PATH)

# Generate data
n = 200
X, y = gen_3_clusters(n)

input_size = size(X, 1)
output_size = size(y, 1)

include("./../AdaBeliefCosAnnealNNTraining.jl")
re, optim_params = network_training(input_size, output_size)
m = re(optim_params)

#Laplace Approximation
la = Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
fit!(la, zip(collect(eachcol(X)), y))
optimize_prior!(la; verbose=true, n_steps=100)

_labels = sort(unique(argmax.(eachcol(y))))
plt_list = []
for target in _labels
    plt = plot(la, X, argmax.(eachcol(y)); target=target, clim=(0, 1), link_approx=:plugin, markersize=2)
    push!(plt_list, plt)
end
plot(plt_list...)
savefig("./plot_showing_plugin_estimator.pdf")

predict_la = LaplaceRedux.predict(la, X, link_approx=:probit)
mapslices(argmax, predict_la, dims=1)
mapslices(x -> 1 - maximum(x), predict_la, dims=1)
entropies = mapslices(x -> normalized_entropy(x, output_size), predict_la, dims=1)

xs = -7.0f0:0.10f0:7.0f0
ys = -7.0f0:0.10f0:7.0f0
heatmap(xs, ys, (x, y) -> normalized_entropy(LaplaceRedux.predict(la, vcat(x, y), link_approx=:probit), output_size))
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./plot_showing_LA_estimator.pdf")
