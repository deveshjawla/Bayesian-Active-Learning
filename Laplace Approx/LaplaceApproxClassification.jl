using Flux #Deep learning
using StatsPlots
using LaplaceRedux

PATH = @__DIR__
cd(PATH)

# Generate data
n = 200
using DataFrames
include("./../DataUtils.jl")
X, Y = gen_3_clusters(n)

input_size = size(X, 1)
output_size = size(Y, 1)

include("./../AdaBeliefCosAnnealNNTraining.jl")
optim_theta, re = network_training("Relu2Layers", input_size, output_size, 100; data=(X, Y), loss_function=Flux.logitcrossentropy)
m = re(optim_theta)

#Laplace Approximation
la = Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
fit!(la, zip(collect(eachcol(X)), Y))

#Using Empirical Bayes, optimise the Prior
optimize_prior!(la; verbose=true, n_steps=100)

predict_la = LaplaceRedux.predict(la, X, link_approx=:probit)
mapslices(argmax, predict_la, dims=1)
mapslices(x -> 1 - maximum(x), predict_la, dims=1)

xs = -7.0f0:0.10f0:7.0f0
ys = -7.0f0:0.10f0:7.0f0
heatmap(xs, ys, (x1,x2) -> LaplaceRedux.predict(la, vcat(x1, x2), link_approx=:probit)[3])
scatter!(X[1, Y[1, :].==1], X[2, Y[1, :].==1], color=:red, label="1")
scatter!(X[1, Y[2, :].==1], X[2, Y[2, :].==1], color=:green, label="2")
scatter!(X[1, Y[3, :].==1], X[2, Y[3, :].==1], color=:blue, label="3")
savefig("./plot_showing_LA_estimator.pdf")
