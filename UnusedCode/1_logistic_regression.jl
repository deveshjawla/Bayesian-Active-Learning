# Function to split samples.
function train_validate_test(df; v=0.6, t=0.8)
    r = size(df, 1)
    val_index = Int(round(r * v))
    test_index = Int(round(r * t))
    train = df[1:val_index, :]
    validate = df[(val_index+1):test_index, :]
    test = df[(test_index+1):end, :]
    return train, validate, test
end

### 
### Data
### 
PATH = @__DIR__
using DataFrames, DelimitedFiles, Statistics, CSV
df = CSV.read(PATH * "/stroke_dataset_categorised.csv", DataFrame)
df = select(df, Not(:id))

using Random
df = df[shuffle(axes(df, 1)), :]
train, validate, test = train_validate_test(df)

###
### Class Balancing
###

function data_balancing(data_xy; balancing=true)
    if balancing == true
        normal_data = data_xy[data_xy[:, end].==0.0, :]
        anomaly = data_xy[data_xy[:, end].==1.0, :]
        data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    else
        nothing
    end
    data_x = Matrix(data_xy)[:, 1:end-1]
    data_y = Int.(data_xy[:, end])
    return data_x, data_y
end

train_x, train_y = data_balancing(train, balancing=false)
validate_x, validate_y = data_balancing(validate, balancing=false)
test_x, test_y = data_balancing(test, balancing=false)

# # #Depending on the experiment we can rejoin train and validate sets
train_x = vcat(train_x, validate_x)
train_y = vcat(train_y, validate_y)


using Turing
using StatsFuns: logistic

# Retrieve the number of observations.
num_obs, num_features = size(train_x)

σ = sqrt(2 / num_features)

# Bayesian logistic regression (LR)
@model function logistic_regression(x, y, num_obs, σ)
    intercept ~ Normal(0, σ)

    θ ~ MvNormal(zeros(num_features), σ .* ones(num_features))

    for i in 1:num_obs
        v = logistic(intercept + sum(θ .* x[i, :]))
        y[i] ~ Bernoulli(v)
    end
end

# Sample using HMC.
m = logistic_regression(train_x, train_y, num_obs, σ)
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