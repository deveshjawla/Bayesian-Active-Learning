num_mcsteps = 1000
datasets = ["banknote", "coalminequakes", "stroke"]
acq_functions = ["TopKBayesian"]
acquisition_sizes = [20]
n_bins = 10

using DataFrames
using CSV
using DelimitedFiles
using Random
using StatsBase
using Distances

include("./CalibrationUtils.jl")

for dataset in datasets
    experiment_name = "001_comparing_different_acq_funcs"
    PATH = @__DIR__
    cd(PATH * "/DataSets/$(dataset)_dataset")

	###
    ### Data
    ###

    PATH = pwd()

	for acquisition_size in acquisition_sizes
		for acq_func in acq_functions
			pipeline_name = "$(acq_func)_$(acquisition_size)_with_$(num_mcsteps)_MCsteps"
			predictions = readdlm("./$(experiment_name)/$(pipeline_name)/predictions/14.csv", ',')
			target_set = CSV.read("test.csv", DataFrame, header=1)
			target_labels = Int.(target_set.label)
			labels = unique(target_labels)
			n_labels = lastindex(labels)
			for label in labels
				indices_ = findall(==(label), Int.(predictions[1,:]))
				true_labels = target_labels[indices_]
				predicted_probs = predictions[2, indices_]
				predicted_labels = Int.(predictions[1, indices_])
				calibration_plot_maker(n_labels, label, n_bins, predicted_probs, true_labels, predicted_labels, experiment_name, pipeline_name)
			end
		end
	end
end