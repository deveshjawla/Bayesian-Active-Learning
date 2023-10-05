"""
For the binning we assume here that the range of values is 0.0-1.0, that each bin closes right, that each bin would have a different size. input is the number of bins, predicted_probs of the predictions, true labels
"""
function conf_bin_indices_bayesian(n_labels, n_bins, predicted_probs, true_labels, predictions)::DataFrame
    bins = Dict{Int,Vector}()
    df = DataFrame(Bin=Int[], BinSize=Int[], BinAccuracy=Float32[], BinConfidence=Float32[], CalibrationGap=Float32[])

	n = n_bins # Specify the number of bins
	interval = (1/n_labels, 1) # Specify the interval
	bin_width = (interval[2] - interval[1])/n  # Calculate the width of each bin
    for i = 1:n_bins
        lower = interval[1]+(bin_width*(i-1))
        upper = interval[1]+(bin_width*i)
        # println(lower, upper)
		bin_size, mean_acc_, mean_conf_, calibration_gap = 0, 0, 0, 0
        bin = findall(x -> x > lower && x <= upper, predicted_probs)
		bin_size = length(bin)
        bins[i] = bin
        if length(predictions[bin]) > 1
            mean_acc_ = mean(true_labels[bin] .== predictions[bin])
            mean_conf_ = mean(predicted_probs[bin])
        else
            mean_acc_ = NaN
            mean_conf_ = NaN
        end
        # println(length(predictions[bin]), ' ', mean_acc_)
        calibration_gap = abs(mean_acc_ - mean_conf_)
		push!(df, [i bin_size mean_acc_ mean_conf_ calibration_gap])
    end
    return df
end

# n = 3 # specify the number of bins
# interval = (0,1) # specify the interval
# bin_width = (interval[2] - interval[1])/n  # calculate the width of each bin
# lower_end_bins = [interval[1]+(bin_width*(i-1)) for i in 1:n] # create the bins using the bin_width
# upper_end_bins = [interval[1]+(bin_width*i) for i in 1:n] # create the bins using the bin_width

using Distributions
using Optim

function ece_mce(df::DataFrame, total_samples)
    ece = sum(filter(!isnan, df.BinSize .* df.CalibrationGap)) / total_samples
    mce = maximum(filter(!isnan, df.CalibrationGap))
    return ece, mce
end

using Gadfly, Cairo, Fontconfig, DataFrames, CSV
width = 8inch
height = 6inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none)
Gadfly.push_theme(theme)
function calibration_plot_maker(n_labels, label, n_bins, predicted_probs, true_labels, predictions, experiment_name, pipeline_name)
    df = conf_bin_indices_bayesian(n_labels, n_bins, predicted_probs, true_labels, predictions)
    total_samples = lastindex(predicted_probs)
    ECE, MCE = ece_mce(df, total_samples)

    writedlm("./Experiments/$(experiment_name)/ece_mce_for_label_$(label)_later_stage.txt", [["ECE", "MCE"] [ECE, MCE]], ',')

	reliability_diagram = plot(df, layer(x=:BinConfidence, y=:BinAccuracy, color=[colorant"blue"],  Geom.point, Geom.line),  layer(x->x, 1/n_labels, 1, color=[colorant"red"]), Guide.xlabel("Confidence"), Guide.ylabel("Accuracy"), Guide.title("$(n_bins) bins, ECE:$(ECE), MCE:$(MCE)"), Coord.cartesian(xmin=1/n_labels, xmax=1.0, ymin=0.0, ymax=1.0))
    reliability_diagram |> PNG("./Experiments/$(experiment_name)/reliability_diagram_for_$(label)_later_stage.png")
end


# Step to Calibration
# 1. Split Data into Train and Test
# 2. Plot Calibration on Train/Test Data and then Calibrate on Train Data
# 3. Plot Calibration on Test Data after calibration

# Logistic function for a scalar input:
function platt(predicted_probs::Float32)
    1.0 / (1.0 + exp(-predicted_probs))
end

function platt(predicted_probs)
    1.0 ./ (1.0 .+ exp.(-predicted_probs))
end

# pred_conf and labels are on the dataset which we use for calibration
function _loss_binary(a, b, pred_conf, labels_)
	if 2 in unique(labels_)
		labels= deepcopy(labels_)
		labels[labels.==2] .= 0
	end
    return -sum(labels .* log.(platt.(pred_conf .* a .+ b)) + (1.0 .- labels) .* log.(1.0 .- platt.(pred_conf .* a .+ b)))
end

# loss((a, b)) = _loss_binary(a, b, pŷ_validate, validate_y)
# @time result = optimize(loss, [1.0, 1.0], LBFGS())
# a, b = result.minimizer
# calibrated_pŷ_test = platt(pŷ_test .* a .+ b)
# println(calibrated_pŷ_test[1:5])
# writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/calibration_step/calibration_fit_params.csv", [a, b], ',')

# calibration_plot_maker(label, n_bins, calibrated_pŷ_test, test_y, "calibration_step/Test/Calibrated", pipeline_name)
