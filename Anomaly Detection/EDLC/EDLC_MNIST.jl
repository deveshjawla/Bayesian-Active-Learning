using Distributed
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)
@everywhere begin
    using EvidentialFlux
    using Flux
    using Plots
    PATH = @__DIR__
    cd(PATH)

    using MLDatasets

    # Generate data
    trainset = MNIST(:train)
    testset = MNIST(:test)

    X_train, y_train = trainset[1:6000]
    X_train = Flux.unsqueeze(X_train, 3)
    y_train = Flux.onehotbatch(y_train, 0:9)
    X_test, y_test = testset[:]
    X_test = Flux.unsqueeze(X_test, 3)
    y_test = Flux.onehotbatch(y_test, 0:9)


    input_size = size(X_train)[1]
    output_size = size(y_train)[1]




    _, re = Flux.destructure(m)
    num_params = 2_088

end
using Statistics


param_matrix, elapsed = ensemble_training(num_params, input_size, output_size, 100, (X_train, y_train); ensemble_size=7, n_epochs=500)


# Test predictions
ŷ_u = pred_analyzer_multiclass(re, X_test, param_matrix)

using StatisticalMeasures: accuracy
println(accuracy(ŷ_u[1, :], Flux.onecold(y_test)))