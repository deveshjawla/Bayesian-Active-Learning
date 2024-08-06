using Clustering
using StatsPlots
using CSV, DataFrames, Printf
using Statistics
using Flux, ParameterSchedulers
using ParameterSchedulers: Scheduler
using Distances

include("computeRBFBetas.jl")
include("computeCentroids.jl")
include("computeRBFActivations.jl")
include("computeWeights.jl")
include("trainRBF.jl")
include("testRBF.jl")
include("./../DataUtils.jl")

# Generate data
n = 200
X, y = gen_3_clusters(n)
X = permutedims(X)
y = Flux.onecold(y)

gridSize = 100
u = range(-10.0f0, 10.0f0, length=gridSize)
v = range(-10.0f0, 10.0f0, length=gridSize)

(Centers, betas, Theta, nn) = trainRbf(X, y, 10, true, 3)

# # ================================================
# #       Evaluate the RBFNN over the grid.
# # ================================================
# # Define a grid over which to evaluate the RBFN.
# # We'll store the scores for each category as well as the 'prediction' for
# # each point on the grid.
########################################
# Draw contour
heatmap(u, v, (x, y) -> testRbf(Centers, betas, Theta, nn, hcat(x, y))[3])
scatter!(Centers[:, 1], Centers[:, 2], color=:blue, label="Centers")
scatter!(X[y.==1, 1], X[y.==1, 2], color=:red, label="1")
scatter!(X[y.==2, 1], X[y.==2, 2], color=:green, label="2")
scatter!(X[y.==3, 1], X[y.==3, 2], color=:blue, label="2")
savefig("./example.pdf")