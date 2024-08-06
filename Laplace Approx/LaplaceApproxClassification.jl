using Flux #Deep learning
using StatsPlots
using LaplaceRedux

PATH = @__DIR__
cd(PATH)

# Generate data
n = 20
using DataFrames
include("./../DataUtils.jl")
train_x, train_y = gen_3_clusters(n)

n_input = size(train_x, 1)
n_output = size(train_y, 1)

include("./../AdaBeliefCosAnnealNNTraining.jl")
optim_theta, re = network_training("Relu2Layers", n_input, n_output, 100; data=(train_x, train_y), loss_function=Flux.logitcrossentropy)

m = re(optim_theta)

#Laplace Approximation
la = Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
fit!(la, zip(collect(eachcol(train_x)), train_y))

#Using Empirical Bayes, optimise the Prior
optimize_prior!(la; verbose=true, n_steps=100)

predict_la = LaplaceRedux.predict(la, train_x, link_approx=:probit)
ŷ_uncertainty = pred_analyzer_multiclass(predict_la, n_output)

X1 = -4.0f0:0.10f0:4.0f0
X2 = -3.0f0:0.10f0:5.0f0
test_x_area = pairs_to_matrix(X1, X2)
predict_la = LaplaceRedux.predict(la, test_x_area, link_approx=:probit)
ŷ_uncertainty = pred_analyzer_multiclass(predict_la, n_output)
uncertainties = reshape(ŷ_uncertainty[2, :], (lastindex(X1), lastindex(X2)))
gr(size=(700, 600), dpi=300)
heatmap(X1, X2, uncertainties)
scatter!(train_x[1, train_y[1, :].==1], train_x[2, train_y[1, :].==1], color=:red, label="1")
scatter!(train_x[1, train_y[2, :].==1], train_x[2, train_y[2, :].==1], color=:green, label="2")
scatter!(train_x[1, train_y[3, :].==1], train_x[2, train_y[3, :].==1], color=:blue, label="3", legend_title="Classes", aspect_ratio=:equal, xlim=[-4, 4], ylim=[-3, 5], colorbar_title="Laplace Approximation Uncertainty")
savefig("./LA.pdf")
