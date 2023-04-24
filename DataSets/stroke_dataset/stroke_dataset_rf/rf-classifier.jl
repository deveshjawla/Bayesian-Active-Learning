include("../TreeEnsemble.jl")
using .TreeEnsemble
using Plots

# using Revise
# call during debugging:
# Revise.track("Utilities.jl")
# Revise.track("Classifier.jl")
# Revise.track("DecisionTree.jl")
# Revise.track("RandomForest.jl")

using DelimitedFiles, DataFrames
using Printf
using Statistics
using Random
using CSV

PATH = @__DIR__
cd(PATH)

name = "balanced_shapley_uncalibrated_10_bins"
mkpath("./experiments/$(name)")

## -------------- load data  -------------- ##
train_xy = CSV.read("./train.csv", DataFrame, header=1)
train_xy[train_xy.stroke.==0, :stroke] .= 2
shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
train_xy = select(train_xy, vcat(shap_importances.feature_name[1:6], "stroke"))

# using MLJ: partition

# train_xy, validate_xy = partition(train_xy, 0.8, shuffle=true, rng=1334)

function data_balancing(data_xy; balancing=true)
    if balancing == true
        normal_data = data_xy[data_xy[:, end].==2.0, :]
        anomaly = data_xy[data_xy[:, end].==1.0, :]
        data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    else
        nothing
    end
    data_x = data_xy[:, 1:end-1]
    data_y = select(data_xy, :stroke)
    return data_x, data_y
end

train_x, train_y = data_balancing(train_xy, balancing=true)
# validate_x, validate_y = data_balancing(validate_xy, balancing=true)

## -------------- fit data  -------------- ##
# dtc = DecisionTreeClassifier(random_state=42)

max_features = ceil(sqrt(size(train_x)[2]))
n_trees = 300
min_samples_leaf = 3
rfc = RandomForestClassifier(random_state=42, n_trees=n_trees, bootstrap=true, oob_score=true, max_features=max_features, min_samples_leaf=min_samples_leaf)

classifier = rfc

# # A handy helper function to rescale our dataset.
# function standardize(x, mean_, std_)
#     return (x .- mean_) ./ (std_ .+ 0.000001)
# end

# train_max = mean(Matrix(train_x), dims=1)
# train_min = std(Matrix(train_x), dims=1)

# train_x = standardize(Matrix(train_x), train_max, train_min)
# test_x = standardize(Matrix(test_x), train_max, train_min)

# train = hcat(train_x, Matrix(train_y))

# train_x = DataFrame(train_x, :auto)
# train_y = DataFrame([train_y], [:target])

elapsed = @elapsed fit!(classifier, train_x, train_y)
print("fitting time  $elapsed ");


# print("prediction time (train)");
# @time acc_train = score(classifier, train_x, train_y);
# print("prediction time (test) ");
# @time acc_test = score(classifier, test_x, test_y);


# println()
# @printf("train accuracy: %.2f%%\n", acc_train * 100)
# if hasproperty(classifier, :oob_score_) && !isnothing(classifier.oob_score_)
#     @printf("obb accuracy:   %.2f%%\n", classifier.oob_score_ * 100)
# end
# @printf("test accuracy:  %.2f%%\n", acc_test * 100)

# println()
# nleaves_ = nleaves(classifier)
# if hasproperty(classifier, :trees)
#     max_depths = [get_max_depth(tree) for tree in classifier.trees]
# else
#     max_depths = [get_max_depth(classifier)]
# end
# @printf("nleaves range, average: %d-%d, %.2f\n",
#     minimum(nleaves_), maximum(nleaves_), mean(nleaves_))
# @printf("max depth range, average: %d-%d, %.2f\n",
#     minimum(max_depths), maximum(max_depths), mean(max_depths))

test_xy = CSV.read("./test.csv", DataFrame, header=1)
test_xy[test_xy.stroke.==-0, :stroke] .= 2
test_xy = select(test_xy, vcat(shap_importances.feature_name[1:6], "stroke"))
test_x, test_y = data_balancing(test_xy, balancing=true)


threshold = 0.5
ŷ = TreeEnsemble.predict(classifier, test_x)

predicted_probs = predict_prob(classifier, test_x)
using EvalMetrics

test_y = Matrix(test_y)[:, 1]
test_y[test_y.==2] .= 0
ŷ[ŷ.==2] .= 0
using Flux: mse

f1 = f1_score(test_y, ŷ)
brier = mse(test_y, predicted_probs[:, 1])
using Plots;
gr();
prplot(test_y, predicted_probs[:, 1])
no_skill(x) = count(==(1), test_y) / length(test_y)
no_skill_score = no_skill(0)
plot!(no_skill, 0, 1, label="No Skill Classifier:$no_skill_score")
savefig("./experiments/$(name)/PRCurve.png")
# mcc = matthews_correlation_coefficient(test_y, ŷ)
# acc = accuracy(ŷ, test_y)
fpr = false_positive_rate(test_y, ŷ)
# fnr = fnr(ŷ, test_y)
# tpr = tpr(ŷ, test_y)
# tnr = tnr(ŷ, test_y)
prec = precision(test_y, ŷ)
# recall = true_positive_rate(ŷ, test_y)
prauc = au_prcurve(test_y, predicted_probs[:, 1])

writedlm("./experiments/$(name)/results.txt", [["threshold", "brier", "f1", "fpr", "precision", "PRAUC"] [threshold, brier, f1, fpr, prec, prauc]], ',')


####
#### Calibration
####

number_of_bins = 10

function conf_bin_indices(n, conf, test, predictions)
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
        if length(predictions[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = count(==(1), test[bin]) / length(test[bin]) #mean(test[bin] .== predictions[bin])
        else
            mean_conf_ = NaN
            mean_acc_ = NaN
        end
        println(length(predictions[bin]), ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end


function ece_mce(bins, calibration_gaps, total_samples)
    n_bins = length(bins)
    ece_ = []
    for i in 1:n_bins
        append!(ece_, length(bins[i]) * calibration_gaps[i])
    end
    ece = sum(filter(!isnan, ece_)) / total_samples
    mce = maximum(filter(!isnan, collect(values(calibration_gaps))))
    return ece, mce
end

# using Distributions
# using Optim

# # Logistic function for a scalar input:
# function platt(conf::Float64)
#     1.0 / (1.0 + exp(-conf))
# end

# function platt(conf)
#     1.0 ./ (1.0 .+ exp.(-conf))
# end

# labels = validate_y.stroke
# ŷ_validate = TreeEnsemble.predict(classifier, validate_x)
# labels[labels.==2] .= 0
# ŷ_validate[ŷ_validate.==2] .= 0
# predicted_probs_validate = predict_prob(classifier, validate_x)
# pred_conf = predicted_probs_validate[:, 1]

# loss((a, b)) = -sum(labels[i] * log(platt(pred_conf[i] * a + b)) + (1.0 - labels[i]) * log(1.0 - platt(pred_conf[i] * a + b)) for i = 1:lastindex(pred_conf))

# @time result = optimize(loss, [1.0, 1.0], LBFGS())

# a, b = result.minimizer

#calibrating the results of test set

# calibrated_pred_prob = platt(predicted_probs[:, 1] .* a .+ b)
calibrated_pred_prob = predicted_probs[:, 1]

# Calibrated Results

bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, calibrated_pred_prob, test_y, ŷ)

total_samples = lastindex(calibrated_pred_prob)

ECE, MCE = ece_mce(bins, calibration_gaps, total_samples)

writedlm("./experiments/$(name)/validate_ece_mce.txt", [ECE, MCE])

f(x) = x
using Plots
reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false, title="Reliability diagram with \n ECE:$(ECE), MCE:$(MCE)",
    xlabel="Confidence",
    ylabel="# Class labels in Target", size=(800, 600))
plot!(f, 0, 1, label="Perfect Calibration")
savefig(reliability_diagram, "./experiments/$(name)/reliability_diagram.png")

# println()
# # confusion matrix
# y_actual = test_y[:, 1]
# y_pred = predict(classifier, test_x)
# C = confusion_matrix(y_actual, y_pred)
# println("Confusion matrix:")
# display(C);
# if size(C, 1) == 2
#     recall, prec, f1 = calc_f1_score(C)
#     @printf("recall, precision, F1: %.2f%%, %.2f%%, %.4f\n",
#         recall * 100, prec * 100, f1)
# end

# println()
# fi1 = classifier.feature_importances
# print("perm_feature_importance time")
# # @time fi_perm = perm_feature_importance(classifier, train_x, train_y, n_repeats=10, random_state=classifier.random_state)
# # fi2 = fi_perm[:means]
# order = sortperm(fi1, rev=true)
# # println("Feature importances")
# # for col_val in zip(names(train_x)[order], fi1[order])
# #     col, val = col_val
# #     @printf("%-15s %.4f\n", col, val)
# # end

# # order = sortperm(fi2)
# # fi2 ./= sum(fi2)

# width = 0.4
# # yticks2 = (1+width/2):1:(length(fi2)+width/2)
# yticks = (1-width/2):1:(length(fi1)-width/2)

# # b1 = bar(yticks2, fi2[order], label="permutation", orientation=:horizontal, bar_width=width, yerr=fi_perm[:stds])
# bar!(yticks, fi1[order], label="impurity", orientation=:horizontal, bar_width=width)
# plot!(
#     yticks=(1:classifier.n_features, classifier.features[order]),
#     xlabel="relative feature importance score",
#     ylabel="feature",
#     title="Feature importances",
#     label="impurity", legend=(0.85, 0.8)
# )

# display(b1)
# #savefig(b1, "UniversalBank_feature_importances_jl")

# # println()
# # print_tree(rfc.trees[1])

# thrity_imp_features = X[:, order[1:30]]

# seventy_imp_features = X[:, order[1:70]]

# writedlm("./secom_data/thrity_imp_features.csv", thrity_imp_features, ',')

# writedlm("./secom_data/seventy_imp_features.csv", seventy_imp_features, ',')
