# 1. Data Split
# 	a. Diversity sampling to obtain the initial dataset
# 2. Define functional Model
# 	a. Define the bayesian Model
# 3. LOOP
# 	a. Set prior
# 	b. condition model on acquired samples 
# 		1. Log performance of newly trained model(include batch size)
# 	c. Score the pool
# 	d. Acquire new samples and add to train set & delete from pool
# 	e. Make poterior as the new prior
# 	f. If new data points are to be added to the pool
# 		1. add the new data to the left over pool
# 		2. Re-normalize the whole train+pool dataset, and then apply the same normalization params to the test set
using DataFrames
using CSV
using DelimitedFiles
using Random
Random.seed!(1234)
using Distances
using CategoricalArrays
using ProgressMeter, Printf
using StatsBase: countmap
using Clustering
using Gadfly, Cairo, Fontconfig, DataFrames, CSV

using Flux
using StatsPlots
using LaplaceRedux
include("./Evidential Deep Learning/DirichletLayer.jl")
# Generate data
include("./DataUtils.jl")
include("./AdaBeliefCosAnnealNNTraining.jl")
include("./NonBayesianALUtils.jl")
include("./NonBayesianInference.jl")
include("./NonBayesianQuery.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

include("./RBFNN/computeRBFBetas.jl")
include("./RBFNN/computeCentroids.jl")
include("./RBFNN/computeRBFActivations.jl")
include("./RBFNN/computeWeights.jl")
include("./RBFNN/trainRBF.jl")
include("./RBFNN/testRBF.jl")

variable_of_comparison = :AcquisitionFunction
x_variable = :CumulativeTrainedSize


list_learning_algorithms = ["LaplaceApprox"] #"RBF", "Evidential", "LaplaceApprox"

acq_functions = ["Initial", "Entropy", "Uncertainty"]

datasets = ["adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps", "iris1988", "yeast1996"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
list_maximum_pool_size = [60, 40, 40, 80, 100, 30, 296] #40, 60, 40, 40, 80, 100, 30, 296
acquisition_sizes = round.(Int, list_maximum_pool_size ./ 5)
# acquisition_sizes = [20, 20, 10, 10, 20, 20, 10, 40]#20, 20, 10, 10, 20, 20, 10, 40 
list_acq_steps = repeat([5], 8)

list_inout_dims = [(4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_n_folds = [5, 5, 5, 5, 3, 5, 5]#5, 5, 5, 5, 5, 3, 5, 5

for learning_algorithm in list_learning_algorithms
    experiment = "ComparisonAcquisitionFunctions$(learning_algorithm)"

    for (dataset, inout_dims, acquisition_size, n_folds, n_acq_steps) in zip(datasets, list_inout_dims, acquisition_sizes, list_n_folds, list_acq_steps)
        # for (dataset, inout_dims, acquisition_size) in zip(datasets, list_inout_dims, acquisition_sizes)
        println(dataset)
        PATH = @__DIR__
        cd(PATH * "/Data/Tabular/$(dataset)")

        n_input, n_output = inout_dims

        ###
        ### Data
        ###

        PATH = pwd()

        df_folds = DataFrame()
        for fold in 1:n_folds
            train = CSV.read("./FiveFolds/train_$(fold).csv", DataFrame, header=1)
            # train = CSV.read("./train.csv", DataFrame, header=1)

            test = CSV.read("./FiveFolds/test_$(fold).csv", DataFrame, header=1)
            # test = CSV.read("./test.csv", DataFrame, header=1)

            pool, test = pool_test_to_matrix(train, test, n_input, learning_algorithm; output_size=n_output)
            total_pool_samples = size(pool[1], 2)

            # println("The number of input features are $n_input")
            # println("The number of outputs are $n_output")
            if n_output == 1
                kpi_df = Array{Any}(missing, 0, 7)
            else
                class_names = Array{String}(undef, n_output)
                for i = 1:n_output
                    class_names[i] = "$(i)"
                end
                kpi_df = Array{Any}(missing, 0, 9 + 2 * n_output)
            end

            for acq_func in acq_functions
                pipeline_name = "$(acquisition_size)_$(acq_func)_$(fold)"
                mkpath("./Experiments/$(experiment)/$(pipeline_name)/classification_performance")
                mkpath("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics")
                mkpath("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions")

                n_acq_steps = running_active_learning(n_acq_steps, pool, n_input, n_output, test, experiment, pipeline_name, acquisition_size, acq_func, learning_algorithm)
                if n_output == 1
                    kpi = collecting_stats_active_learning_experiments_regression(n_acq_steps, experiment, pipeline_name)
                else
                    kpi = collecting_stats_active_learning_experiments_classification(n_acq_steps, acq_func, experiment, pipeline_name, n_output)
                end
                # kpi = readdlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", ',')
                kpi_df = vcat(kpi_df, permutedims(kpi))
            end
            if n_output == 1
                kpi_names = vcat([:AcquisitionSize, :MSE, :MAE, :AcquisitionFunction, :Experiment, :CumulativeTrainedSize], :Elapsed)
            elseif n_output == 2
                kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :F1Score, :AcquisitionFunction, :Experiment, :CumulativeTrainedSize], Symbol.(class_names), Symbol.(class_names), :CumCDE, :Elapsed)
            else
                kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :AcquisitionFunction, :Experiment, :CumulativeTrainedSize], Symbol.(class_names), Symbol.(class_names), :CumCDE, :Elapsed)
            end

            df_fold = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df_fold)
            df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            if n_output == 1
                auc_per_fold(fold, df_fold, variable_of_comparison, :MSE, :Elapsed, experiment)
            else
                auc_per_fold(fold, df_fold, variable_of_comparison, :BalancedAccuracy, :Elapsed, experiment)
            end
            df_folds = vcat(df_folds, df_fold)
        end
        CSV.write("./Experiments/$(experiment)/df_folds.csv", df_folds)

        if n_output == 1
            auc_mean(n_folds, experiment, variable_of_comparison, :MSE, :Elapsed)
        else
            auc_mean(n_folds, experiment, variable_of_comparison, :BalancedAccuracy, :Elapsed)
        end

        df_folds = CSV.read("./Experiments/$(experiment)/df_folds.csv", DataFrame)
        if n_output == 1
            list_plotting_measurables = [:MSE, :Elapsed]
            list_plotting_measurables_mean = [:MSE_mean, :Elapsed_mean]
            list_plotting_measurables_std = [:MSE_std, :Elapsed_std]
            list_normalised_or_not = [false, false]
        elseif n_output == 2
            list_plotting_measurables = [:BalancedAccuracy, :F1Score, :Elapsed]
            list_plotting_measurables_mean = [:BalancedAccuracy_mean, :F1Score_mean, :Elapsed_mean]
            list_plotting_measurables_std = [:BalancedAccuracy_std, :F1Score_std, :Elapsed_std]
            list_normalised_or_not = [false, false, false]
        else
            list_plotting_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed]
            list_plotting_measurables_mean = [:BalancedAccuracy_mean, :AverageClassAccuracyHarmonicMean_mean, :Elapsed_mean]
            list_plotting_measurables_std = [:BalancedAccuracy_std, :AverageClassAccuracyHarmonicMean_std, :Elapsed_std]
            list_normalised_or_not = [false, false, false]
        end


        mean_std_by_group(df_folds, variable_of_comparison, x_variable, experiment; list_measurables=list_plotting_measurables)
        for (i, j, k, l) in zip(list_plotting_measurables, list_plotting_measurables_mean, list_plotting_measurables_std, list_normalised_or_not)
            plotting_measurable_variable(experiment, variable_of_comparison, acq_functions, dataset, x_variable, i, j, k, l)
        end
    end
end