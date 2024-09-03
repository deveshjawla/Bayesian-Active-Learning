datasets = ["stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps", "iris1988", "yeast1996"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
list_inout_dims = [(4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_learning_algorithms = ["RBF", "Evidential", "LaplaceApprox", "MCMC"]

using DelimitedFiles, DataFrames, CSV, Statistics

for (dataset, inout_dims) in zip(datasets, list_inout_dims)
    PATH = @__DIR__
    cd(PATH * "/Data/Tabular/$(dataset)/Experiments")
    println(dataset)

    n_input, n_output = inout_dims

    if n_output == 1
        list_measurables = [:MSE, :Elapsed]
    elseif n_output == 2
        list_measurables = [:BalancedAccuracy, :F1Score, :Elapsed]
    else
        list_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed]
    end

    col_names = vcat([:Algorithm, :AcquisitionFunction], list_measurables)
    convergence_stats = Matrix{Any}(undef, 0, lastindex(col_names))
    for learning_algorithm in list_learning_algorithms
        if learning_algorithm == "MCMC"
            experiment = "ComparisonBayesianAcquisitionFunctions$(learning_algorithm)_ReverseF1"
            acq_functions = ["Random", "BALD", "StdConfidence", "BayesianUncertainty0.8"] # "Random", "Random",  "BALD", "StdConfidence", "BayesianUncertainty0.8"
        else
            experiment = "ComparisonAcquisitionFunctions$(learning_algorithm)_Weighted"
            acq_functions = ["Random", "Entropy", "Uncertainty"]
        end

        array_measurables = Array{String}(undef, lastindex(acq_functions), lastindex(list_measurables))
        for (idx, measurable) in enumerate(list_measurables)
            mean_std = CSV.read("./$(experiment)/mean_auc_$(measurable).csv", DataFrame)

            stats_df_avg = mean_std[!, Symbol("$(measurable)_mean")]
            stats_df_std = mean_std[!, Symbol("$(measurable)_confidence_interval_95")]

            stats_df_ = []
            for (i, j) in zip(stats_df_avg, stats_df_std)
                stats_df_ = vcat(stats_df_, join([string(round(i, digits=2)), " ± ", round(j, digits=2)]))
            end
            array_measurables[:, idx] = stats_df_
        end
		array_per_algorithm = hcat(acq_functions, array_measurables)
		convergence_stats = vcat(convergence_stats, hcat(repeat([learning_algorithm], lastindex(acq_functions)), array_per_algorithm))
    end
	CSV.write("./Comparison of Models and Acquisition Functions.csv", DataFrame(convergence_stats, col_names))
end