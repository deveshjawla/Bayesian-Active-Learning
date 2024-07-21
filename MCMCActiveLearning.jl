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
num_chains = 2

experiments = ["WeightedLikelihood"]
datasets = ["iris1988", "yeast1996"]#20, 20, 10, 10, 20, 20, 10, 40
# acquisition_sizes = [20, 20, 10, 10, 20, 20, 10, 40]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
minimum_training_sizes = [30, 296] #60, 60, 40, 40, 80, 100, 30, 296
acquisition_sizes = round.(Int, minimum_training_sizes ./ 2)
list_acq_steps = repeat([1], 8)

list_inout_dims = [(4, 3), (8, 10)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_n_folds = [5, 5]#5, 5, 5, 5, 5, 3, 5, 5

list_class_balancing = ["UnBalancedAcquisition"] #"BalancedBinaryAcquisition"
list_prior_informativeness = ["UnInformedPrior"] # "UnInformedPrior", "InformedPrior", "NoInit"
list_prior_variance = ["GlorotPrior"] # "GlorotPrior", 0.01, 0.2, 1.0, 3.0, 5.0
list_likelihood_name = ["WeightedLikelihood"] #"UnWeightedLikelihood", "WeightedLikelihood", "Regression"
acq_functions = ["Initial"] # "BayesianUncertainty", "Initial", "Random"
# temperature = nothing, or a Float or list of nothing and Floats, nothing invokes a non-customised Likelihood in the @model
temperatures = [nothing] # 1.0, 0.1, 0.001 or nothing

# Add four processes to use for sampling.
addprocs(num_chains; exeflags=`--project`)

using DataFrames
using CSV
using DelimitedFiles
using Random
Random.seed!(1234);
using StatsBase
using Distances
using CategoricalArrays
using ProgressMeter
@everywhere using StatsBase
using Gadfly, Cairo, Fontconfig, DataFrames, CSV
width = 6inch
height = 6inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
Gadfly.push_theme(theme)

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
            total_pool_samples = size(pool[1])[2]

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
                                    # using Zygote
                                    # Turing.setadbackend(:zygote)
                                    using ReverseDiff
                                    Turing.setadbackend(:reversediff)

                                    #Here we define the Prior
                                    if prior_variance isa Number
                                        prior_std = prior_variance .* ones(num_params)
                                    else
                                        # GlorotNormal initialisation
                                        prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + n_output)) * ones(n_output_layer))
                                        # prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
                                    end
                                end

                                num_mcsteps = 1000
                                pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(fold)_$(num_chains)_$(num_mcsteps)"
                                # pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(num_chains)_$(num_mcsteps)"
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/predictions")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/hyperpriors")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/classification_performance")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/independent_param_matrix_all_chains")
                                # mkpath("./Experiments/$(experiment)/$(pipeline_name)/log_distribution_changes")
                                mkpath("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions")

                                running_active_learning_ensemble!(num_params, prior_std, pool, n_input, n_output, test, experiment, acquisition_size, num_mcsteps, num_chains, temperature, prior_informativeness, prior_variance, likelihood_name)
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
                kpi_names = vcat([:AcquisitionSize, :MSE, :MAE, :AcquisitionFunction, :Temperature, :Experiment, :CumTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName], :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS)
            else
                kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :Accuracy, :EnsembleMajority, :AcquisitionFunction, :Temperature, :Experiment, :CumTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName, :F1], Symbol.(class_names), Symbol.(class_names), :CumCDE, :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS)
            end

            df = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df)
            if n_output == 1
                auc_per_fold!(fold, df, :PriorInformativeness, :MSE, :Elapsed)
            else
                auc_per_fold!(fold, df, :AcquisitionFunction, :Accuracy, :Elapsed)
            end
            # df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            df_folds = vcat(df_folds, df_fold)
        end
        CSV.write("./Experiments/$(experiment)/df_folds.csv", df_folds)
        if n_output == 1
            auc_mean(n_folds, experiment, :PriorInformativeness, :MSE, :Elapsed)
            mean_std_by_group(df_folds, :LikelihoodName, :PriorVariance; list_measurables=[:MSE, :Elapsed, :AcceptanceRate, :NumericalErrors])
        else
            auc_mean(n_folds, experiment, :AcquisitionFunction, :Accuracy, :Elapsed)
            mean_std_by_group(df_folds, :AcquisitionFunction, :CumTrainedSize; list_measurables=[:Accuracy, :Elapsed, :EnsembleMajority, :AcceptanceRate, :NumericalErrors])
        end

        ## Loops when using Cross Validation for AL
        if n_output == 1
            list_plotting_measurables = [:MSE, :Elapsed]
            list_plotting_measurables_mean = [:MSE_mean, :Elapsed_mean]
            list_plotting_measurables_std = [:MSE_std, :Elapsed_std]
            for (i, j, k) in zip(list_plotting_measurables, list_plotting_measurables_mean, list_plotting_measurables_std)
                plotting_measurable_variable(experiment, acq_functions, dataset, :CumTrainedSize, i, j, k)
            end
        else
            list_plotting_measurables = [:Accuracy, :F1, :Elapsed]
            list_plotting_measurables_mean = [:Accuracy_mean, :F1_mean, :Elapsed_mean]
            list_plotting_measurables_std = [:Accuracy_std, :F1_std, :Elapsed_std]
            for (i, j, k) in zip(list_plotting_measurables, list_plotting_measurables_mean, list_plotting_measurables_std)
                plotting_measurable_variable(experiment, acq_functions, dataset, :CumTrainedSize, i, j, k)
            end
        end
    end
end