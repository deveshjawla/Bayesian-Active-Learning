using DelimitedFiles, DataFrames, CSV, Statistics
using StatsPlots
# using Plots: text

function plotter(cr::Matrix{Float32}, sr::Matrix{Float32}, cols::Vector{Symbol}, path_name)::Nothing
    (n, m) = size(cr)
    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="Pearson Correlation Coefficients")
    annotate!([(j, i, text("$(round(cr[i, j], digits=2)) ± $(round(sr[i, j], digits=2))", 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])
    savefig("$(path_name)/pearson_correlations_Uncertainties.png")
    return nothing
end

let

    datasets = ["stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps", "iris1988"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"

    list_maximum_pool_size = 2 .* [40, 60, 40, 40, 80, 100, 30] #40, 60, 40, 40, 80, 100, 30, 296
    acquisition_sizes = round.(Int, list_maximum_pool_size ./ 2)
    # acquisition_sizes = [40, 60, 40, 40, 80, 100, 30] #40, 60, 40, 40, 80, 100, 30, 296

    list_acq_steps = repeat([1], 8)

    # list_acq_steps = [10, 10, 10, 10, 10, 10, 5] # 10, 10, 10, 10, 10, 10, 5, 5

    list_inout_dims = [(4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

    # list_n_folds = repeat([1], 8)
    list_n_folds = [10, 10, 10, 10, 10, 10, 5]

    PATH = @__DIR__
    convergence_stats = Array{Float32}(undef, 5, 5, 0)
    for (dataset, inout_dims, acquisition_size, n_folds, n_acq_steps) in zip(datasets, list_inout_dims, acquisition_sizes, list_n_folds, list_acq_steps)
        cd(PATH * "/Data/Tabular/$(dataset)/Experiments")
        println(dataset)

		for fold in n_folds
        	M = readdlm("./Correlations_uncertainty/$(acquisition_size)_Random_GlorotPrior_WeightedLikelihood_UnInformedPrior_CWL_$(fold)_8_200/classification_performance/corr_matrix_1.csv", ',')
        	convergence_stats = cat(convergence_stats, M, dims=3)
		end
    end
    mean_stats = mean(convergence_stats, dims=3)
    std_stats = std(convergence_stats, dims=3)

    cols = [:Confidence, :StdDeviationConfidence, :Aleatoric, :Epistemic, :TotalUncertainty]
    plotter(mean_stats[:, :, 1], std_stats[:, :, 1], cols, PATH)
    writedlm("$(PATH)/Correlations_uncertainty.csv", convergence_stats[:, :, 1], ',')
end