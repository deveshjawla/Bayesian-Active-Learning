using Turing
using FillArrays
using Flux
using Plots
using ReverseDiff

using LinearAlgebra
using Random

# Use reverse_diff due to the number of parameters in neural networks.
Turing.setadbackend(:reversediff)

# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i in 1:M]))

x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i in 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2 * M); zeros(2 * M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1, y1; color="red", clim=(0, 1))
    return Plots.scatter!(x2, y2; color="blue", clim=(0, 1))
end

plot_data()

# Construct a neural network using Flux
nn_initial = Chain(Dense(2, 3, relu), Dense(3, 2, relu), Dense(2, 1, σ))

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(nn_initial)

lastindex(parameters_initial) # number of paraemters in NN

mutable struct BernoulliNew <: DiscreteUnivariateDistribution
    pred
end

struct BernoulliN <: DiscreteUnivariateDistribution
    preds::Array
	function BernoulliNew(preds::Array)
		return Bernoulli(preds)
	end
end

BernoulliN(pred) = Bernoulli(pred)

using Parameters
import Distributions: logpdf
function logpdf(dist::BernoulliNew, data::Array)
	@unpack preds = dist
    LL = 0.0
    for i in 1:lastindex(data)
        LL += logpdf(Bernoulli(pred[i]), data[i])
    end
    return LL/0.01
end


@model function bayes_nn(xs, ts, nparameters, reconstruct; alpha=0.01)
    # Create the weight and bias vector.
    parameters ~ MvNormal(zeros(nparameters), I / alpha)

    # Construct NN from parameters
    nn = reconstruct(parameters)
    # Forward NN to make predictions
    preds = nn(xs)

    # Observe each prediction.
    ts ~ BernoulliNew(preds)
end

using Turing
# Perform inference.
N = 100
ch = sample(bayes_nn(hcat(xs...), ts, lastindex(parameters_initial), reconstruct), Turing.NUTS(), N)


using Turing,Parameters

import Distributions: logpdf
mutable struct mydist{T1,T2} <: ContinuousUnivariateDistribution
    μ::T1
    σ::T2
end

function logpdf(dist::mydist,data::Array{Float32,1})
    @unpack μ,σ=dist
    LL = 0.0
    for d in data
        LL += logpdf(Normal(μ,σ),d)
    end
    return LL
end

@model function model2(y)
    μ ~ Normal(0,1)
    σ ~ Truncated(Cauchy(0,1),0,Inf)
    y ~ mydist(μ,σ)
end

data = rand(Normal(0,1),1000)

chain2 = sample(model2(data), NUTS(), 10)
describe(chain2)