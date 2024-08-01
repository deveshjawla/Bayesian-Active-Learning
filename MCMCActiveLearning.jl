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

using Distributed
using Turing
num_chains = 8
# Add four processes to use for sampling.
addprocs(num_chains; exeflags=`--project`)

variable_of_comparison = :AcquisitionFunction
x_variable = :CumulativeTrainedSize
datasets = ["stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
list_maximum_pool_size = [40, 120, 40, 40, 80, 100, 30, 296] #40, 120, 40, 40, 80, 100, 30, 296
acquisition_sizes = round.(Int, list_maximum_pool_size ./ 5)
# acquisition_sizes = [20, 20, 10, 10, 20, 20, 10, 40]#20, 20, 10, 10, 20, 20, 10, 40 
list_acq_steps = repeat([5], 8)

list_inout_dims = [(4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_n_folds = [5, 5, 5, 5, 5, 3, 5, 5]#5, 5, 5, 5, 5, 3, 5, 5

num_mcsteps = 1000
experiments = ["ActiveLearning"]


list_prior_informativeness = ["UnInformedPrior"] # "UnInformedPrior", "InformedPrior", "NoInit"
list_prior_variance = ["GlorotPrior"] # "GlorotPrior", 0.01, 0.2, 1.0, 3.0, 5.0
list_likelihood_name = ["WeightedLikelihood"] #"UnWeightedLikelihood", "WeightedLikelihood", "Regression"
acq_functions = ["Initial", "BALD", "StdConfidence", "BayesianUncertainty0.8"] # "BayesianUncertainty0.8", "Initial", "Random"
# temperature = nothing, or a Float or list of nothing and Floats, nothing invokes a non-customised Likelihood in the @model
temperatures = ["CWL"] # 1.0, 0.1, 0.001 or "CWL"

using DataFrames
using CSV
using DelimitedFiles
using Random
Random.seed!(1234)
using StatsBase
using Distances
using CategoricalArrays
using ProgressMeter
@everywhere using StatsBase
using Gadfly, Cairo, Fontconfig, DataFrames, CSV

include("./MCMCUtils.jl")
include("./Query.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

for experiment in experiments
    @everywhere experiment = $experiment
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
        @everywhere PATH = $PATH

        df_folds = DataFrame()
        for fold in 1:n_folds
            train = CSV.read("./FiveFolds/train_$(fold).csv", DataFrame, header=1)
            # train = CSV.read("./train.csv", DataFrame, header=1)

            test = CSV.read("./FiveFolds/test_$(fold).csv", DataFrame, header=1)
            # test = CSV.read("./test.csv", DataFrame, header=1)

            pool, test = pool_test_to_matrix(train, test, n_input, "MCMC")
            total_pool_samples = size(pool[1], 2)

            # println("The number of input features are $n_input")
            # println("The number of outputs are $n_output")
            if n_output == 1
                kpi_df = Array{Any}(missing, 0, 15)
            else
                class_names = Array{String}(undef, n_output)
                for i = 1:n_output
                    class_names[i] = "$(i)"
                end

                kpi_df = Array{Any}(missing, 0, 18 + 2 * n_output)
            end
            for prior_informativeness in list_prior_informativeness
                @everywhere prior_informativeness = $prior_informativeness
                for prior_variance in list_prior_variance
                    @everywhere prior_variance = $prior_variance
                    for likelihood_name in list_likelihood_name
                        @everywhere likelihood_name = $likelihood_name
                        for acq_func in acq_functions
                            @everywhere acq_func = $acq_func
                            for temperature in temperatures
                                @everywhere temperature = $temperature

                                @everywhere begin
                                    n_input = $n_input
                                    n_output = $n_output
                                    using Flux, Turing

                                    ###
                                    ### Dense Network specifications(Functional Model)
                                    ###
                                    include(PATH * "/Network.jl")
                                    include("./BayesianModel.jl")

                                    # setprogress!(false)


                                    using ReverseDiff


                                    #Here we define the Prior
                                    if prior_variance isa Number
                                        prior_std = prior_variance .* ones(num_params)
                                    else
                                        # GlorotNormal initialisation
                                        prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + n_output)) * ones(n_output_layer))
                                        # prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
                                    end
                                end


                                pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(fold)_$(num_chains)_$(num_mcsteps)"
                                # pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(num_chains)_$(num_mcsteps)"
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/predictions")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/hyperpriors")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/classification_performance")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/independent_param_matrix_all_chains")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/log_distribution_changes")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions")

                                n_acq_steps = running_active_learning_ensemble(n_acq_steps, num_params, prior_std, pool, n_input, n_output, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, temperature, prior_informativeness, likelihood_name)
                                if n_output == 1
                                    kpi = collecting_stats_active_learning_experiments_regression(n_acq_steps, experiment, pipeline_name, num_chains, n_output)
                                else
                                    kpi = collecting_stats_active_learning_experiments_classification(n_acq_steps, experiment, pipeline_name, num_chains, n_output)
                                end
                                # kpi = readdlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", ',')
                                kpi_df = vcat(kpi_df, permutedims(kpi))
                            end
                        end
                    end
                end
            end
            if n_output == 1
                kpi_names = vcat([:AcquisitionSize, :MSE, :MAE, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName], :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS, :MaxPSRF)
			elseif n_output == 2
                kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :F1Score, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName], Symbol.(class_names), Symbol.(class_names), :CumCDE, :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS, :MaxPSRF)
			else
                kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName], Symbol.(class_names), Symbol.(class_names), :CumCDE, :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS, :MaxPSRF)
            end

            df_fold = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df_fold)
            df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            if n_output == 1
                auc_per_fold(fold, df_fold, variable_of_comparison, :MSE, :Elapsed)
            else
                auc_per_fold(fold, df_fold, variable_of_comparison, :BalancedAccuracy, :Elapsed)
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
            list_plotting_measurables = [:BalancedAccuracy, :F1Score, :Elapsed, :AvgESS]
            list_plotting_measurables_mean = [:BalancedAccuracy_mean, :F1Score_mean, :Elapsed_mean, :AvgESS_mean]
            list_plotting_measurables_std = [:BalancedAccuracy_std, :F1Score_std, :Elapsed_std, :AvgESS_std]
            list_normalised_or_not = [false, false, false, false]
		else
			list_plotting_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed, :AvgESS]
            list_plotting_measurables_mean = [:BalancedAccuracy_mean, :AverageClassAccuracyHarmonicMean_mean, :Elapsed_mean, :AvgESS_mean]
            list_plotting_measurables_std = [:BalancedAccuracy_std, :AverageClassAccuracyHarmonicMean_std, :Elapsed_std, :AvgESS_std]
            list_normalised_or_not = [false, false, false, false]
        end

        mean_std_by_group(df_folds, variable_of_comparison, x_variable; list_measurables=list_plotting_measurables)
        for (i, j, k, l) in zip(list_plotting_measurables, list_plotting_measurables_mean, list_plotting_measurables_std, list_normalised_or_not)
            plotting_measurable_variable(experiment, variable_of_comparison, acq_functions, dataset, x_variable, i, j, k, l)
        end
    end
end