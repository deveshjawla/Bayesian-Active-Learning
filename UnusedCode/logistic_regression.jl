# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    train = df[1:index, :]
    test = df[(index+1):end, :]
    return train, test
end

### 
### Data
### 
PATH = @__DIR__
using DataFrames, DelimitedFiles, Statistics
features = readdlm(PATH * "/secom_data_preprocessed_moldovan2017.csv", ',', Float64)
labels = Int.(readdlm(PATH * "/secom_labels.txt")[:, 1])

using Random
data = hcat(features, labels)
data = data[shuffle(axes(data, 1)), :]
train, test = split_data(data, at=0.9)

train_x = train[:, 1:end-1]
train_y = Int.(train[:, end])
train_y[train_y.==-1] .= 0
# train_y = Bool.(train_y)
# train_y = hcat([Flux.onehot(i, [1, 2]) for i in train_y]...)
# train_data = Iterators.repeated((train_x', train_y_onehot), 128)

test_x = test[:, 1:end-1]
test_y = Int.(test[:, end])
test_y[test_y.==-1] .= 0
# test_y = Bool.(test_y)
# test_y = hcat([Flux.onehot(i, [1, 2]) for i in test_y]...)

# A handy helper function to rescale our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

train_mean = mean(train_x, dims=1)
train_std = std(train_x, dims=1)

train_x = standardize(train_x, train_mean, train_std)
test_x = standardize(test_x, train_mean, train_std)

if count(isnan.(test_x)) > 1 || count(isnan.(test_x)) > 1
	error("NaNs found in Data")
end

###
### Class Balancing
###

train = hcat(train_x, train_y)

postive_data = train[train[:, end].==1.0, :]
negative_data = train[train[:, end].==0.0, :]
train = vcat(postive_data, negative_data[1:size(postive_data)[1], :])
# data = data[1:200, :]
train = train[shuffle(axes(train, 1)), :]


train_x = train[:, 1:end-1]
train_y = Int.(train[:, end])

using Turing
using StatsFuns: logistic


# Retrieve the number of observations.
l, n = size(train_x)

σ = sqrt(2 / n)

# Bayesian logistic regression (LR)
@model function logistic_regression(x, y, n, σ)
    intercept ~ Normal(0, σ)

    θ ~ MvNormal(zeros(454), σ .* ones(454))

    for i in 1:n
        v = logistic(intercept + sum(θ .* x[i, :]))
        y[i] ~ Bernoulli(v)
    end
end

# Sample using HMC.
m = logistic_regression(train_x, train_y, l, σ)
chain_timed = @timed sample(m, NUTS(50, 0.65), 100)
chain = chain_timed.value
elapsed = chain_timed.time

# describe(chain)
intercept = chain[:intercept]
θ = MCMCChains.group(chain, :θ).value
intercepts = collect.(eachrow(intercept[:, 1]))
thetas = collect.(eachrow(θ[:, :, 1]))

if sum(map(x->count(isnan.(x)), intercepts)) > 1 || sum(map(x->count(isnan.(x)), thetas)) > 1
    error("NaNs found in Params")
end

using MLJ

experiment = "lr_3"
mkdir("./test_results_$(experiment)")

function pred_analyzer(test_xs, intercepts, thetas, threshold)
    l = length(intercepts)
    # Retrieve the number of rows.
    n, _ = size(test_xs)

    predictions = Array{Float64,2}(undef, n, l)
    for j in 1:l

        # Generate a vector to store our predictions.
        v = Vector{Int}(undef, n)

        # Calculate the logistic function for each element in the test set.
        for i in 1:n
            num = logistic(intercepts[j][1] + sum(thetas[j] .* test_xs[i, :]))
            if num >= threshold
                v[i] = 1
            else
                v[i] = 0
            end
        end
        predictions[:, j] = v
    end

    predictions_mean = mean(predictions, dims=2)
    predictions_std = std(predictions, dims=2)

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return predictions_mean, predictions_std
end

# Set the prediction threshold.
thresholds = collect(range(0.01, 0.99, step=0.01))

for threshold in thresholds
    # # Make the predictions.
    # predictions = prediction(test_x, chain, threshold)

    # # Calculate MSE for our test set.
    # loss = sum((predictions - test_y) .^ 2) / length(test_y)


    # defaults = sum(test_y)
    # not_defaults = length(test_y) - defaults

    # predicted_defaults = sum(test_y .== predictions .== 1)
    # predicted_not_defaults = sum(test_y .== predictions .== 0)

    # println("Defaults: $defaults
    #  Predictions: $predicted_defaults
    #  Percentage defaults correct $(predicted_defaults/defaults)")

    # println("Not defaults: $not_defaults
    #  Predictions: $predicted_not_defaults
    #  Percentage non-defaults correct $(predicted_not_defaults/not_defaults)")
    predictions_mean, predcitions_std = pred_analyzer(test_x, intercepts, thetas, threshold)

	predictions_mean = vec(round.(Int, predictions_mean))

    mcc = MLJ.mcc(predictions_mean, test_y)
    f1 = MLJ.f1score(predictions_mean, test_y)
    acc = MLJ.accuracy(predictions_mean, test_y)
    fpr = MLJ.fpr(predictions_mean, test_y)
    fnr = MLJ.fnr(predictions_mean, test_y)
    tpr = MLJ.tpr(predictions_mean, test_y)
    tnr = MLJ.tnr(predictions_mean, test_y)
    prec = MLJ.precision(predictions_mean, test_y)
    recall = MLJ.recall(predictions_mean, test_y)

    writedlm("./test_results_$(experiment)/results_$(threshold).txt", [elapsed, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall], ',')
end