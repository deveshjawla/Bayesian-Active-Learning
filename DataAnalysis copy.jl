datasets = ["stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps", "iris1988"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
list_inout_dims = [(4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_noise_x = [false, true]
list_noise_y = [false, true]

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

    col_names = vcat([:Noise_X, :Noise_Y], list_measurables)
    convergence_stats = Matrix{Any}(undef, 0, lastindex(col_names))
    for noise_x in list_noise_x
        for noise_y in list_noise_y
            experiment = "NoiseX = $(noise_x) and NoiseY = $(noise_y)"

            array_measurables = Array{String}(undef, 1, lastindex(list_measurables))
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
            array_per_algorithm = hcat([noise_x noise_y], array_measurables)
            convergence_stats = vcat(convergence_stats, array_per_algorithm)
        end
    end
    CSV.write("./Effect of Noise Parameter.csv", DataFrame(convergence_stats, col_names))
end