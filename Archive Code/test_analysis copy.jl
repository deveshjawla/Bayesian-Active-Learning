
PATH = @__DIR__
using DataFrames, DelimitedFiles, Statistics
test_x = readdlm(PATH * "/test_x.csv", ',', Float64)
test_y = Int.(readdlm(PATH * "/test_y.csv")[:, 1])

###
### Dense Network specifications
###

using Flux

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:1350], 9, 150)
    b0 = θ[1351:1359]
    W1 = reshape(θ[1360:1386], 3, 9)
    b1 = θ[1387:1389]
    W2 = reshape(θ[1390:1392], 1, 3)
    b2 = θ[1393:1393]
    model = Chain(
        Dense(W0, b0, tanh),
        Dense(W1, b1, tanh),
        Dense(W2, b2, sigmoid)
    )
    return model
end

params_matrix = readdlm(PATH * "/param_matrix.csv", ',', Float64)
predictions_mean = vec(round.(Int,readdlm(PATH * "/predictions_mean.csv", ',', Float64)))
params_matrix = readdlm(PATH * "/param_matrix.csv", ',', Float64)

using MLJ

for i in [1, 2, 3, 300, 400, 500]
    params = params_matrix[i, :]
    model = feedforward(params)
    ŷ = model(test_x')
    predictions = (ŷ .> 0.5)
    # count(ŷ .> 0.7)
    # count(test_y)

    mcc = MLJ.mcc(predictions, test_y')
    f1 = MLJ.f1score(predictions, test_y')
    acc = MLJ.accuracy(predictions, test_y')
    fpr = MLJ.fpr(predictions, test_y')
    fnr = MLJ.fnr(predictions, test_y')
    tpr = MLJ.tpr(predictions, test_y')
    tnr = MLJ.tnr(predictions, test_y')
    prec = MLJ.precision(predictions, test_y')
    recall = MLJ.recall(predictions, test_y')

    writedlm(PATH * "/performance_net_$(i).txt", [0.0, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall], ',')
end

mcc = MLJ.mcc(predictions_mean, test_y)
f1 = MLJ.f1score(predictions_mean, test_y)
acc = MLJ.accuracy(predictions_mean, test_y)
fpr = MLJ.fpr(predictions_mean, test_y)
fnr = MLJ.fnr(predictions_mean, test_y)
tpr = MLJ.tpr(predictions_mean, test_y)
tnr = MLJ.tnr(predictions_mean, test_y)
prec = MLJ.precision(predictions_mean, test_y)
recall = MLJ.recall(predictions_mean, test_y)

writedlm(PATH * "/results.txt", [0.0, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall], ',')

# using AdvancedVI
# AdvancedVI.elbo(advi, q, m, 1000)