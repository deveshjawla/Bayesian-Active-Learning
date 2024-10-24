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

using DataFrames
using CSV
using DelimitedFiles
using Random
Random.seed!(1234)
using Distances
using CategoricalArrays
using ProgressMeter
@everywhere using StatsBase
@everywhere using Flux
using Gadfly, Cairo, Fontconfig, DataFrames, CSV

include("./MCMCUtils.jl")
include("./MCMCInference.jl")
include("./Query.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")
include("./Variational Inference/VI_Inference.jl")

variable_of_comparison = :AcquisitionFunction
x_variable = :CumulativeTrainedSize
datasets = ["coalmineseismicbumps", "adult1994", "creditdefault2005", "stroke", "banknote2012", "creditfraud", "iris1988", "yeast1996"]#
acquisition_sizes = repeat([10], 8)
list_acq_steps = [10, 10, 10, 10, 10, 10, 5, 10] # 10, 10, 10, 10, 10, 10, 5, 10
list_inout_dims = [(19, 2), (50, 2), (32, 2), (19, 2), (4, 2), (28, 2), (4, 3), (8, 10)] # 
num_categories_list = [[2, 3, 3], [2, 5, 15, 6, 9, 7], [1, 7, 4], [4, 3, 4], [0], [0], [0], [0]]# 
list_continuos_featuers = [11, 6, 20, 8, 4, 28, 4, 8]# 
list_n_folds = [10, 10, 10, 10, 10, 10, 5, 10] # 10, 10, 10, 10, 10, 10, 5, 10

num_mcsteps = 200
list_learning_algorithms = ["MCMC"]

list_prior_informativeness = ["UnInformedPrior"] # "UnInformedPrior", "InformedPrior", "NoInit"
list_prior_variance = [1.0] # "GlorotPrior", 0.01, 0.2, 1.0, 3.0, 5.0
list_likelihood_name = ["WeightedLikelihood"] #"UnWeightedLikelihood", "WeightedLikelihood", "Regression"
acq_functions = ["Random", "BALD", "StdConfidence", "BayesianUncertainty0.8"] # "Random", "BALD", "StdConfidence", "BayesianUncertainty0.8"
# temperature = nothing, or a Float or list of nothing and Floats, nothing invokes a non-customised Likelihood in the @model
temperatures = ["CWL"] # 1.0, 0.1, 0.001 or "CWL"

list_noise_x = [false]
list_noise_y = [false]

for (dataset, num_categories, n_cont_featuers, inout_dims, acquisition_size, n_folds, n_acq_steps) in zip(datasets, num_categories_list, list_continuos_featuers, list_inout_dims, acquisition_sizes, list_n_folds, list_acq_steps)
    for (learning_algorithm, noise_x, noise_y) in Iterators.product(list_learning_algorithms, list_noise_x, list_noise_y)
        experiment = "Test"

        @everywhere experiment = $experiment
        # for (dataset, inout_dims, acquisition_size) in zip(datasets, list_inout_dims, acquisition_sizes)
        println(dataset)
        PATH = @__DIR__
        cd(PATH * "/Data/Tabular/$(dataset)")

        n_input, n_output = inout_dims

        counter = 0
        cat_indices_list = []
        if num_categories[1] == 0
            nothing
        else
            for i in num_categories
                push!(cat_indices_list, counter+1:counter+i)
                counter += i
            end
        end

        continuos_features_indices = counter+1:counter+n_cont_featuers

        push!(cat_indices_list, continuos_features_indices)
        # push!(cat_indices_list, continuos_features_indices)
        # push!(cat_indices_list, continuos_features_indices)

        if n_output == 1
            list_auc_measurables = [:MSE, :Elapsed]
        elseif n_output == 2
            list_auc_measurables = [:BalancedAccuracy, :F1Score, :Elapsed]
        else
            list_auc_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed]
        end

        ###
        ### Data
        ###

        PATH = pwd()
        @everywhere PATH = $PATH

        df_folds = DataFrame()
        for fold in 1:n_folds
            train = CSV.read("./TenFolds/train_$(fold).csv", DataFrame, header=1)
            test = CSV.read("./TenFolds/test_$(fold).csv", DataFrame, header=1)

            # categorical_train_inputs = nothing
            # categorical_test_inputs = nothing
            # try
            #     categorical_train_inputs = CSV.read("./TenFolds/categorical_train_$(fold).csv", DataFrame, header=1)
            # catch
            #     error("categorical_train_inputs = nothing")
            # end
            # # train = CSV.read("./train.csv", DataFrame, header=1)

            # try
            #     categorical_test_inputs = CSV.read("./TenFolds/categorical_test_$(fold).csv", DataFrame, header=1)
            # catch
            #     error("categorical_test_inputs = nothing")
            # end

            pool, test = pool_test_to_matrix(train, test, n_input, "MCMC")
            total_pool_samples = size(pool[1], 2)

            # categories_train = map(unique, eachcol(categorical_train_inputs))
            # categories_test = map(unique, eachcol(categorical_test_inputs))
            # categories_total = map((x, y) -> union(x, y), categories_train, categories_test)

            # num_categories = map(lastindex, categories_total)

            # println(num_categories)

            # categorical_tensor_pool = onehotbatch_indices(collect(eachcol(categorical_train_inputs)), num_categories)
            # categorical_tensor_test = onehotbatch_indices(collect(eachcol(categorical_test_inputs)), num_categories)

            # println("The number of continuos input features are $n_input")
            # println("The number of outputs are $n_output")
            if learning_algorithm == "MCMC"
                if n_output == 1
                    kpi_df = Array{Any}(missing, 0, 15)
                else
                    class_names = Array{String}(undef, n_output)
                    for i ∈ 1:n_output
                        class_names[i] = "$(i)"
                    end
                    kpi_df = Array{Any}(missing, 0, 18 + 2 * n_output)
                end
            elseif learning_algorithm == "VI"
                if n_output == 1
                    kpi_df = Array{Any}(missing, 0, 10)
                else
                    class_names = Array{String}(undef, n_output)
                    for i ∈ 1:n_output
                        class_names[i] = "$(i)"
                    end
                    kpi_df = Array{Any}(missing, 0, 13 + 2 * n_output)
                end
            end


            for (prior_informativeness, prior_variance, likelihood_name, acq_func, temperature) in Iterators.product(list_prior_informativeness, list_prior_variance, list_likelihood_name, acq_functions, temperatures)
                @everywhere prior_informativeness = $prior_informativeness
                @everywhere prior_variance = $prior_variance
                @everywhere likelihood_name = $likelihood_name
                @everywhere acq_func = $acq_func
                @everywhere temperature = $temperature
                @everywhere begin
                    n_input = $n_input
                    n_output = $n_output
                    n_cont_featuers = $n_cont_featuers
                    num_categories = $num_categories

                    cat_indices_list = $cat_indices_list
                    continuos_features_indices = $continuos_features_indices

                    # setprogress!(false)
                    using Flux, Turing
                    using Turing: Variational
                    using ReverseDiff


                    ###
                    ### Dense Network specifications(Functional Model)
                    ###
                    # include(PATH * "/Network.jl")
                    include("./BayesianModel.jl")

                    # Create the model instance
                    model = MultiInputModel(num_categories, n_cont_featuers, n_output)

                    init_params, feedforward = Flux.destructure(model)

                    # # Call the model with the inputs
                    # categorical_tensor_pool = $categorical_tensor_pool
                    # pool = $pool
                    # @show output = model(categorical_tensor_pool, pool[1])

                    num_params = lastindex(init_params)

                    @info "num_params" num_params

                    #Here we define the Prior
                    # if prior_variance isa Number
                    #     prior_std = prior_variance .* ones(num_params)
                    # else
                    #     # GlorotNormal initialisation
                    #     prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + n_output)) * ones(n_output_layer))
                    #     # prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
                    # end

                    prior_std = ones(num_params)
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

                n_acq_steps = running_active_learning_ensemble(
                    n_acq_steps,
                    num_params,
                    prior_std,
                    pool,
                    n_input,
                    n_output,
                    cat_indices_list,
                    test,
                    experiment,
                    pipeline_name,
                    acquisition_size,
                    num_mcsteps,
                    num_chains,
                    temperature,
                    prior_informativeness,
                    likelihood_name,
                    learning_algorithm,
                    noise_x,
                    noise_y,
                )
                if n_output == 1
                    kpi = collecting_stats_active_learning_experiments_regression(n_acq_steps, experiment, pipeline_name, num_chains, learning_algorithm)
                else
                    kpi = collecting_stats_active_learning_experiments_classification(n_acq_steps, experiment, pipeline_name, num_chains, n_output, learning_algorithm)
                end
                kpi = readdlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", ',')
                kpi_df = vcat(kpi_df, permutedims(kpi))
            end
            if learning_algorithm == "MCMC"
                if n_output == 1
                    kpi_names = vcat(
                        [:AcquisitionSize, :MSE, :MAE, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName],
                        :Elapsed,
                        :OOBRhat,
                        :AcceptanceRate,
                        :NumericalErrors,
                        :AvgESS,
                        :MaxPSRF,
                    )
                elseif n_output == 2
                    kpi_names = vcat(
                        [:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :F1Score, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName],
                        Symbol.(class_names),
                        Symbol.(class_names),
                        :CumCDE,
                        :Elapsed,
                        :OOBRhat,
                        :AcceptanceRate,
                        :NumericalErrors,
                        :AvgESS,
                        :MaxPSRF,
                    )
                else
                    kpi_names = vcat(
                        [:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName],
                        Symbol.(class_names),
                        Symbol.(class_names),
                        :CumCDE,
                        :Elapsed,
                        :OOBRhat,
                        :AcceptanceRate,
                        :NumericalErrors,
                        :AvgESS,
                        :MaxPSRF,
                    )
                end
            elseif learning_algorithm == "VI"
                if n_output == 2
                    kpi_names = vcat(
                        [:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :F1Score, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName],
                        Symbol.(class_names),
                        Symbol.(class_names),
                        :CumCDE,
                        :Elapsed,
                    )
                else
                    kpi_names = vcat(
                        [:AcquisitionSize, :ClassDistEntropy, :BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :AcquisitionFunction, :Temperature, :Experiment, :CumulativeTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName],
                        Symbol.(class_names),
                        Symbol.(class_names),
                        :CumCDE,
                        :Elapsed,
                    )
                end
            end

            df_fold = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df_fold)

            df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            for auc_measurable in list_auc_measurables
                auc_per_fold(fold, df_fold, variable_of_comparison, auc_measurable, experiment)
            end

            df_folds = vcat(df_folds, df_fold)
        end
        CSV.write("./Experiments/$(experiment)/df_folds.csv", df_folds)

        for auc_measurable in list_auc_measurables
            auc_mean(n_folds, experiment, variable_of_comparison, auc_measurable)
        end

        df_folds = CSV.read("./Experiments/$(experiment)/df_folds.csv", DataFrame)
        if learning_algorithm == "MCMC"
            if n_output == 1
                list_plotting_measurables = [:MSE, :Elapsed]
                list_plotting_measurables_mean = [:MSE_mean, :Elapsed_mean]
                list_plotting_measurables_confidence_interval_95 = [:MSE_confidence_interval_95, :Elapsed_confidence_interval_95]
                list_normalised_or_not = [false, false]
            elseif n_output == 2
                list_plotting_measurables = [:BalancedAccuracy, :F1Score, :Elapsed, :AvgESS]
                list_plotting_measurables_mean = [:BalancedAccuracy_mean, :F1Score_mean, :Elapsed_mean, :AvgESS_mean]
                list_plotting_measurables_confidence_interval_95 = [:BalancedAccuracy_confidence_interval_95, :F1Score_confidence_interval_95, :Elapsed_confidence_interval_95, :AvgESS_confidence_interval_95]
                list_normalised_or_not = [true, true, false, false]
            else
                list_plotting_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed, :AvgESS]
                list_plotting_measurables_mean = [:BalancedAccuracy_mean, :AverageClassAccuracyHarmonicMean_mean, :Elapsed_mean, :AvgESS_mean]
                list_plotting_measurables_confidence_interval_95 = [:BalancedAccuracy_confidence_interval_95, :AverageClassAccuracyHarmonicMean_confidence_interval_95, :Elapsed_confidence_interval_95, :AvgESS_confidence_interval_95]
                list_normalised_or_not = [true, true, false, false]
            end
        elseif learning_algorithm == "VI"
            if n_output == 1
                list_plotting_measurables = [:MSE, :Elapsed]
                list_plotting_measurables_mean = [:MSE_mean, :Elapsed_mean]
                list_plotting_measurables_confidence_interval_95 = [:MSE_confidence_interval_95, :Elapsed_confidence_interval_95]
                list_normalised_or_not = [false, false]
            elseif n_output == 2
                list_plotting_measurables = [:BalancedAccuracy, :F1Score, :Elapsed]
                list_plotting_measurables_mean = [:BalancedAccuracy_mean, :F1Score_mean, :Elapsed_mean]
                list_plotting_measurables_confidence_interval_95 = [:BalancedAccuracy_confidence_interval_95, :F1Score_confidence_interval_95, :Elapsed_confidence_interval_95]
                list_normalised_or_not = [true, true, false, false]
            else
                list_plotting_measurables = [:BalancedAccuracy, :AverageClassAccuracyHarmonicMean, :Elapsed]
                list_plotting_measurables_mean = [:BalancedAccuracy_mean, :AverageClassAccuracyHarmonicMean_mean, :Elapsed_mean]
                list_plotting_measurables_confidence_interval_95 = [:BalancedAccuracy_confidence_interval_95, :AverageClassAccuracyHarmonicMean_confidence_interval_95, :Elapsed_confidence_interval_95]
                list_normalised_or_not = [true, true, false, false]
            end
        end

        mean_std_by_group(df_folds, variable_of_comparison, x_variable, experiment; list_measurables=list_plotting_measurables)
        for (i, j, k, l) in zip(list_plotting_measurables, list_plotting_measurables_mean, list_plotting_measurables_confidence_interval_95, list_normalised_or_not)
            plotting_measurable_variable(experiment, variable_of_comparison, acq_functions, dataset, x_variable, i, j, k, l)
        end
    end
end
