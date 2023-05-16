# For the binning we assume here that the range of values is 0.0-1.0, that each bin closes right, that each bin would have a different size

function conf_bin_indices(n, conf, test, predictions)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float64}()
    bin_acc = Dict{Int,Float64}()
    calibration_gaps = Dict{Int,Float64}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        if length(predictions[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = mean(test[bin] .== predictions[bin])
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

#input is the number of bins, confidence scores of the predictions, true labels
function conf_bin_indices(n, conf, test)
    bins = Dict{Int,Vector}()
    mean_conf = Dict{Int,Float64}()
    bin_acc = Dict{Int,Float64}()
    calibration_gaps = Dict{Int,Float64}()
    for i in 1:n
        lower = (i - 1) / n
        upper = i / n
        # println(lower, upper)
        bin = findall(x -> x > lower && x <= upper, conf)
        bins[i] = bin
        if lastindex(test[bin]) > 1
            mean_conf_ = mean(conf[bin])
            mean_acc_ = count(==(1), test[bin]) / length(test[bin])
        else
            mean_conf_ = NaN
            mean_acc_ = NaN
        end
        println(lastindex(test[bin]), ' ', mean_acc_)
        mean_conf[i] = mean_conf_
        bin_acc[i] = mean_acc_
        calibration_gaps[i] = abs(mean_acc_ - mean_conf_)
    end
    return bins, mean_conf, bin_acc, calibration_gaps
end

using Distributions
using Optim

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

# Logistic function for a scalar input:
function platt(conf::Float64)
    1.0 / (1.0 + exp(-conf))
end

function platt(conf)
    1.0 ./ (1.0 .+ exp.(-conf))
end

# pred_conf and labels are on the dataset which we use for calibration
function _loss_binary(a, b, pred_conf, labels_)
	if 2 in unique(labels_)
		labels= deepcopy(labels_)
		labels[labels.==2] .= 0
	end
    return -sum(labels .* log.(platt.(pred_conf .* a .+ b)) + (1.0 .- labels) .* log.(1.0 .- platt.(pred_conf .* a .+ b)))
end


function calibration_plot_maker(i, number_of_bins, confidence, ground_truth_, stepname::String, name)
    ground_truth = deepcopy(ground_truth_)
    ground_truth[ground_truth.==2] .= 0
    bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, confidence, ground_truth)

    total_samples = lastindex(confidence)

    ECE, MCE = ece_mce(bins, calibration_gaps, total_samples)

    writedlm("./$(experiment_name)/$(name)/" * stepname * "/ece_mce_$(i).txt", [["ECE", "MCE"] [ECE, MCE]], ',')

    f(x) = x
    reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false, title="Reliability diagram with \n ECE:$(ECE), MCE:$(MCE)",
        xlabel="Confidence",
        ylabel="# Class labels in Target", size=(800, 600))
    plot!(f, 0, 1, label="Perfect Calibration")
    savefig(reliability_diagram, "./$(experiment_name)/$(name)/" * stepname * "/reliability_diagram_$(i).png")
end

# Step to Calibration
# 1. Split Data into Train and Test
# 2. Plot Calibration on Train/Test Data and then Calibrate on Train Data
# 3. Plot Calibration on Test Data after calibration

# loss((a, b)) = _loss_binary(a, b, pŷ_validate, validate_y)
# @time result = optimize(loss, [1.0, 1.0], LBFGS())
# a, b = result.minimizer
# calibrated_pŷ_test = platt(pŷ_test .* a .+ b)
# println(calibrated_pŷ_test[1:5])
# writedlm("./$(experiment_name)/$(name)/calibration_step/calibration_fit_params.csv", [a, b], ',')

# calibration_plot_maker(i, number_of_bins, calibrated_pŷ_test, test_y, "calibration_step/Test/Calibrated", name)
