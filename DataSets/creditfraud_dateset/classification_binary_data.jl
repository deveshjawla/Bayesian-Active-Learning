# # Function to split samples.
# function train_validate_test(df; v=0.6, t=0.8)
#     r = size(df, 1)
#     val_index = Int(round(r * v))
#     test_index = Int(round(r * t))
#     train = df[1:val_index, :]
#     validate = df[(val_index+1):test_index, :]
#     test = df[(test_index+1):end, :]
#     return train, validate, test
# end

# ### 
# ### Data
# ### 
# PATH = @__DIR__
# using DataFrames, DelimitedFiles, Statistics
# features = readdlm(PATH * "/secom_data_preprocessed_moldovan2017.csv", ',', Float64)
# labels = Int.(readdlm(PATH * "/secom_labels.txt")[:, 1])
# labels[labels.==-1] .= 0

# using Random
# data = hcat(features, labels)
# data = data[shuffle(axes(data, 1)), :]
# train, validate, test = train_validate_test(data)

# ###
# ### Class Balancing
# ###

# function data_balancing(data_xy; balancing=true)
#     if balancing == true
#         normal_data = data_xy[data_xy[:, end].==0.0, :]
#         anomaly = data_xy[data_xy[:, end].==1.0, :]
#         data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
#         data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
#     else
#         nothing
#     end
#     data_x = data_xy[:, 1:end-1]
#     data_y = Int.(data_xy[:, end])
#     return data_x, data_y
# end

# train_x, train_y = data_balancing(train, balancing=false)
# validate_x, validate_y = data_balancing(validate, balancing=false)
# test_x, test_y = data_balancing(test, balancing=false)

# # # #Depending on the experiment we can rejoin train and validate sets
# train_x = vcat(train_x, validate_x)
# train_y = vcat(train_y, validate_y)

# # A handy helper function to rescale our dataset.
# function standardize(x, mean_, std_)
#     return (x .- mean_) ./ (std_ .+ 0.000001)
# end

# train_mean = mean(train_x, dims=1)
# train_std = std(train_x, dims=1)

# train_x = standardize(train_x, train_mean, train_std)
# validate_x = standardize(validate_x, train_mean, train_std)
# test_x = standardize(test_x, train_mean, train_std)


# # # using MultivariateStats

# # # M = fit(PCA, train_x', maxoutdim=150)
# # # train_x_transformed = MultivariateStats.transform(M, train_x')

# # # # M = fit(PCA, test_x', maxoutdim = 150)
# # # test_x_transformed = MultivariateStats.transform(M, test_x')

# # # train_x = train_x_transformed'
# # # test_x = test_x_transformed'


# name = "data_unbalanced"

# mkdir("./$(experiment_name)/$(name)")
# mkdir("./$(experiment_name)/$(name)/DATA")
# writedlm("./$(experiment_name)/$(name)/DATA/train_y.csv", train_y, ',')
# writedlm("./$(experiment_name)/$(name)/DATA/train_x.csv", train_x, ',')
# writedlm("./$(experiment_name)/$(name)/DATA/validate_y.csv", validate_y, ',')
# writedlm("./$(experiment_name)/$(name)/DATA/validate_x.csv", validate_x, ',')
# writedlm("./$(experiment_name)/$(name)/DATA/test_y.csv", test_y, ',')
# writedlm("./$(experiment_name)/$(name)/DATA/test_x.csv", test_x, ',')

#reading back data
name = "test_unbalanced"
using DelimitedFiles
train_y = vec(readdlm("./$(experiment_name)/$(name)/DATA/train_y.csv", ',', Int))
train_x = readdlm("./$(experiment_name)/$(name)/DATA/train_x.csv", ',')
validate_y = vec(readdlm("./$(experiment_name)/$(name)/DATA/validate_y.csv", ',', Int))
validate_x = readdlm("./$(experiment_name)/$(name)/DATA/validate_x.csv", ',')
using MLJ
test_y = vec(readdlm("./$(experiment_name)/$(name)/DATA/test_y.csv", ',', Int))
test_x = readdlm("./$(experiment_name)/$(name)/DATA/test_x.csv", ',')
name = "test_unbalanced_relu"
mkdir("./$(experiment_name)/$(name)")


###
### Dense Network specifications
###

input_size = size(train_x)[2]
l1, l2, l3, l4, l5 = 100, 100, 20, 20, 1
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
ol5 = l4 * l5 + l5

total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

using Flux

function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:45400], 100, 454)
    b0 = θ[45401:45500]
    W1 = reshape(θ[45501:55500], 100, 100)
    b1 = θ[55501:55600]
    W2 = reshape(θ[55601:57600], 20, 100)
    b2 = θ[57601:57620]
    W3 = reshape(θ[57621:58020], 20, 20)
    b3 = θ[58021:58040]
    W4 = reshape(θ[58041:58060], 1, 20)
    b4 = θ[58061:58061]
    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        Dense(W2, b2, relu),
        Dense(W3, b3, relu),
        Dense(W4, b4, sigmoid)
    )
    return model
end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:15000], 100, 150)
#     b0 = θ[15001:15100]
#     W1 = reshape(θ[15101:25100], 100, 100)
#     b1 = θ[25101:25200]
#     W2 = reshape(θ[25201:27200], 20, 100)
#     b2 = θ[27201:27220]
#     W3 = reshape(θ[27221:27620], 20, 20)
#     b3 = θ[27621:27640]
#     W4 = reshape(θ[27641:27660], 1, 20)
#     b4 = θ[27661:27661]
#     model = Chain(
#         Dense(W0, b0, leakyrelu),
#         Dense(W1, b1, leakyrelu),
#         Dense(W2, b2, leakyrelu),
#         Dense(W3, b3, leakyrelu),
#         Dense(W4, b4, σ)
#     )
#     return model
# end

###
### Bayesian Network specifications
###

using Turing
# using Zygote
# Turing.setadbackend(:zygote)
using ReverseDiff
Turing.setadbackend(:reversediff)

# alpha = 0.09
# sigma = sqrt(1.0 / alpha)

#Here we define the layer by layer initialisation
sigma = vcat(sqrt(2 / input_size) * ones(nl1), sqrt(2 / l1) * ones(nl2), sqrt(2 / l2) * ones(nl3), sqrt(2 / l3) * ones(nl4), sqrt(2 / l4) * ones(ol5))

@model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(total_num_params), sigma)
    nn = feedforward(θ)
    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Bernoulli(ŷ[i])
    end
end

###
### Inference
###

# ScikitLearn.CrossValidation.StratifiedKFold([ones(10)...,zeros(5)...], n_folds=5)

chain_timed = @timed sample(bayesnn(Array(train_x'), train_y), NUTS(50, 0.65), 1000)
chain = chain_timed.value
elapsed = chain_timed.time
θ = MCMCChains.group(chain, :θ).value

params_set = collect.(eachrow(θ[:, :, 1]))

param_matrix = mapreduce(permutedims, vcat, params_set)
mkdir("./$(experiment_name)/$(name)/BNN")
writedlm("./$(experiment_name)/$(name)/BNN/param_matrix.csv", param_matrix, ',')

function pred_analyzer(test_xs, test_ys, params_set)::Tuple{Array{Float32},Array{Float32},Array{Int},Array{Int},Array{Float32}}
    means = []
    stds = []
    classifications = []
    majority_voting = []
    majority_conf = []
    for (test_x, test_y) in zip(eachrow(test_xs), test_ys)
        predictions = []
        for theta in params_set
            model = feedforward(theta)
            ŷ = model(collect(test_x))
            append!(predictions, ŷ)
        end
        individual_classifications = round.(Int, predictions)
        majority_vote = ifelse(mean(individual_classifications) > 0.5, 1, 0)
        majority_conf_ = mean(individual_classifications)
        ensemble_pred_prob = mean(predictions) #average logit
        std_pred_prob = std(predictions)
        ensemble_class = round(Int, ensemble_pred_prob)
        append!(means, ensemble_pred_prob)
        append!(stds, std_pred_prob)
        append!(classifications, ensemble_class)
        append!(majority_voting, majority_vote)
        append!(majority_conf, majority_conf_)
    end

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return means, stds, classifications, majority_voting, majority_conf
end

validate_x = test_x
validate_y = test_y

predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = pred_analyzer(validate_x, validate_y, params_set)

writedlm("./$(experiment_name)/$(name)/BNN/predcitions_std.csv", predcitions_std, ',')
writedlm("./$(experiment_name)/$(name)/BNN/predictions_mean.csv", predictions_mean, ',')
writedlm("./$(experiment_name)/$(name)/BNN/classifications.csv", classifications, ',')
writedlm("./$(experiment_name)/$(name)/BNN/majority_vote.csv", majority_vote, ',')
writedlm("./$(experiment_name)/$(name)/BNN/majority_conf.csv", majority_conf, ',')

# Reading back results
predcitions_std = vec(readdlm("./$(experiment_name)/$(name)/BNN/predcitions_std.csv", ','))
predictions_mean = vec(readdlm("./$(experiment_name)/$(name)/BNN/predictions_mean.csv", ','))
classifications = coerce(vec(readdlm("./$(experiment_name)/$(name)/BNN/classifications.csv", ',', Int)), OrderedFactor)
majority_vote = coerce(vec(readdlm("./$(experiment_name)/$(name)/BNN/majority_vote.csv", ',', Int)), OrderedFactor)
majority_conf = vec(readdlm("./$(experiment_name)/$(name)/BNN/majority_conf.csv", ','))

validate_y_labels = coerce(Int.(validate_y), OrderedFactor)
levels!(validate_y_labels, [0, 1])
levels!(classifications, [0, 1])

using MLJ

# for i in [1, 10, 20, 80, 90, 100]
#     params = param_matrix[i, :]
#     model = feedforward(params)
#     ŷ = model(test_x')
#     predictions = round.(Int, ŷ)

#     mcc = MLJ.mcc(predictions, test_y')
#     f1 = MLJ.f1score(predictions, test_y')
#     acc = MLJ.accuracy(predictions, test_y')
#     fpr = MLJ.fpr(predictions, test_y')
#     fnr = MLJ.fnr(predictions, test_y')
#     tpr = MLJ.tpr(predictions, test_y')
#     tnr = MLJ.tnr(predictions, test_y')
#     prec = MLJ.precision(predictions, test_y')
#     recall = MLJ.recall(predictions, test_y')

#     writedlm("./$(experiment_name)/$(name)/performance_net_$(i).txt", [0.0, mcc, f1, acc, fpr, fnr, tpr, tnr, prec, recall], ',')
# end

mcc = MLJ.mcc(validate_y_labels, classifications)
brier = MLJ.rmse(validate_y, predictions_mean)^2 #using mse because unidimensional probability
f2score = FScore(β=2.0)
f1 = MLJ.f1score(classifications, validate_y_labels)
f2 = f2score(classifications, validate_y_labels)
acc = MLJ.accuracy(classifications, validate_y_labels)
fpr = MLJ.fpr(classifications, validate_y_labels)
fnr = MLJ.fnr(classifications, validate_y_labels)
tpr = MLJ.tpr(classifications, validate_y_labels)
tnr = MLJ.tnr(classifications, validate_y_labels)
prec = MLJ.precision(classifications, validate_y_labels)
recall = MLJ.recall(classifications, validate_y_labels)

writedlm("./$(experiment_name)/$(name)/BNN/validate_results.txt", [["elapsed", "mcc", "brier", "f1", "f2", "acc", "fpr", "fnr", "tpr", "tnr", "prec", "recall"] [elapsed, mcc, brier, f1, f2, acc, fpr, fnr, tpr, tnr, prec, recall]], ',')

####
#### Calibration
####

number_of_bins = 10

function conf_bin_indices(n, conf, test, predictions)
    predictions = MLJ.coerce(predictions, OrderedFactor)
    test = MLJ.coerce(test, OrderedFactor)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float32}()
    bin_acc = Dict{Int,Float32}()
    calibration_gaps = Dict{Int,Float32}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        mean_conf_ = mean(conf[bin])
        if lastindex(predictions[bin]) > 1
            mean_acc_ = MLJ.truepositive(predictions[bin], test[bin]) / lastindex(test[bin]) #mean(test[bin] .== predictions[bin])
        else
            mean_acc_ = NaN
        end
        println(test[bin], ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end

bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, predictions_mean, validate_y, classifications)


function ece_mce(bins, calibration_gaps, total_samples)
    n_bins = lastindex(bins)
    ece_ = []
    for i in 1:n_bins
        append!(ece_, lastindex(bins[i]) * calibration_gaps[i])
    end
    ece = sum(filter(!isnan, ece_)) / total_samples
    mce = maximum(filter(!isnan, collect(values(calibration_gaps))))
    return ece, mce
end

total_samples = lastindex(predictions_mean)

ECE, MCE = ece_mce(bins, calibration_gaps, total_samples)

writedlm("./$(experiment_name)/$(name)/BNN/validate_ece_mce.txt", [ECE, MCE])

using Plots
reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false, title="Reliability diagram with \n ECE:$(ECE), MCE:$(MCE) \n MCC:$(mcc) and BrierScore:$(brier)",
    xlabel="Confidence",
    ylabel="Accuracy", size=(800, 800))
savefig(reliability_diagram, "./$(experiment_name)/$(name)/BNN/validate_reliability_diagram.png")


# using Distributions
# using Optim

# # Logistic function for a scalar input:
# function platt(conf::Float64)
#     1.0 / (1.0 + exp(-conf))
# end

# function platt(conf)
#     1.0 ./ (1.0 .+ exp.(-conf))
# end

# labels = validate_y
# predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = pred_analyzer(validate_x, validate_y, params_set)
# pred_conf = predictions_mean

# loss((a, b)) = -sum(labels[i] * log(platt(pred_conf[i] * a + b)) + (1.0 - labels[i]) * log(1.0 - platt(pred_conf[i] * a + b)) for i = 1:lastindex(pred_conf))

# @time result = optimize(loss, [1.0, 1.0], LBFGS())

# a, b = result.minimizer


# #calibrating the results of test set

# number_of_bins = 10

# predictions_mean, predcitions_std, classifications, majority_vote, majority_conf = pred_analyzer(test_x, test_y, params_set)

# bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, predictions_mean, test_y, classifications)

# reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false)
# savefig(reliability_diagram, "./$(experiment_name)/$(name)/BNN/test_reliability_diagram.png")

# total_samples = lastindex(predictions_mean)

# ECE = ece(bins, calibration_gaps, total_samples)
# MCE = mce(calibration_gaps)

# writedlm("./$(experiment_name)/$(name)/BNN/test_ece_mce.txt", [ECE, MCE])

# calibrated_pred_prob = platt(predictions_mean .* -a .- b)
# classifications = round.(Int, calibrated_pred_prob)

# bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, calibrated_pred_prob, test_y, classifications)

# reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false)
# savefig(reliability_diagram, "./$(experiment_name)/$(name)/BNN/calibrated_test_reliability_diagram.png")

# total_samples = lastindex(calibrated_pred_prob)

# ECE = ece(bins, calibration_gaps, total_samples)
# MCE = mce(calibration_gaps)

# writedlm("./$(experiment_name)/$(name)/BNN/calibrated_test_ece_mce.txt", [ECE, MCE])