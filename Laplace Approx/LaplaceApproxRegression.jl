using Flux, Plots
using Distributions: Normal, Product
using Statistics
using DelimitedFiles, DataFrames
using LaplaceRedux
PATH = @__DIR__
cd(PATH)

experiment = "DeepEnsembleWithGlorotNormal"
activation_function = "mixed"
function_names = ["Cosine"]#  "Cosine", "Polynomial", "Exponential", "Logarithmic","sin(pisin)"

# load dependencies
using ProgressMeter
using CSV

using Flux
include("./../Functional Prior - Simple Regression Tasks/MakeData.jl")
include("./../AdaBeliefCosAnnealNNTraining.jl")

for function_name in function_names
    
	xs, ys, Xline = make_data(function_name)

    X = [[i] for i in xs]
    data = zip(X, ys)
    train_x = hcat(X...)
    train_y = permutedims(ys)
    n_input, n_output = 1, 1
    m = nn_without_dropout(n_input, n_output)
    params_, rec = Flux.destructure(m)
    num_params = lastindex(params_)
	trained_params, re = network_training(m, n_input, n_output, n_epochs, train_loader, sample_weights_loader)

    plt = 0

    nn = rec(trained_params)
    subset_w = :all
    la = Laplace(nn; likelihood=:regression, subset_of_weights=subset_w)
    # data = Flux.DataLoader((train_x, train_y), batchsize=1)
    fit!(la, data)
    optimize_prior!(la)
    preds = predict(la, permutedims(Xline), link_approx=:mc)

	fμ, fvar = la(_x)
	fμ = vec(fμ)
	fσ = vec(sqrt.(fvar))
	pred_std = sqrt.(fσ .^ 2 .+ la.σ^2)
	plot!(
		x_range,
		fμ;
		color=2,
		label="yhat",
		ribbon=(1.96 * pred_std, 1.96 * pred_std),
		lw=lw,
		kwargs...
	)   # the specific values 1.96 are used here to create a 95% confidence interval
	
    plt = plot(la, Xline, preds, fillalpha=0.7, ylim=[-2, 2], legend=:none, label="Laplace Approximation", fmt=:pdf, size=(600, 400), dpi=300)

    plot!(Xline, map(x -> f(x, function_name), Xline), label="Truth", color=:green)
    scatter!(xs, ys, color=:green, label="Training data", markerstrokecolor=:green)

    mkpath("./$(experiment)")
    savefig(plt, "./$(experiment)/$(function_name)_$(activation_function).pdf")
end