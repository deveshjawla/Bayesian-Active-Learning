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

function gendata(n)
    x1 = randn(Float32, 2, n)
    x2 = randn(Float32, 2, n) .+ [2, 2]
    x3 = randn(Float32, 2, n) .+ [-2, 2]
    y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
    y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
    y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
    hcat(x1, x2, x3), permutedims(hcat(y1, y2, y3))
end

# Generate data
n = 200
X, y = gendata(n)
X = permutedims(X)
y = Flux.onecold(y)

gridSize = 100
u = range(-10.0f0, 10.0f0, length=gridSize)
v = range(-10.0f0, 10.0f0, length=gridSize)

#count = 100
#t = zeros(count, 1)
#for timingIndex = 1:count
#	tic();
(Centers, betas, Theta, nn) = trainRbf(X, y, 10, false)
#	t[timingIndex] = toc();
#end

########################################
# Let's test
model = nn(Theta)

# # ================================================
# #       Evaluate the RBFNN over the grid.
# # ================================================
# # Define a grid over which to evaluate the RBFN.
# # We'll store the scores for each category as well as the 'prediction' for
# # each point on the grid.
########################################
# Draw contour
heatmap(u, v, (x, y) -> softmax(model(getRBFActivations(Centers, betas, hcat(x, y))))[3])
scatter!(Centers[:, 1], Centers[:, 2], color=:blue, label="Centers")
scatter!(X[y.==1, 1], X[y.==1, 2], color=:red, label="1")
scatter!(X[y.==2, 1], X[y.==2, 2], color=:green, label="2")
scatter!(X[y.==3, 1], X[y.==3, 2], color=:blue, label="2")
savefig("./example.pdf")