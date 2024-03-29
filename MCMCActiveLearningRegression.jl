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
num_chains = 3

experiments = ["001"]
datasets = ["TrainingTimeBNN"]#10
# acquisition_sizes = [10]#"TrainingTimes"
acquisition_sizes = [11]

list_inout_dims = [(2, 1)] #

list_n_folds = [1]#5

list_prior_informativeness = ["UnInformedPrior"] # "UnInformedPrior", "InformedPrior", "NoInit"
list_prior_variance = [1.0] # "GlorotPrior", 0.01, 0.2, 1.0, 3.0, 5.0
list_likelihood_name = ["Regression"] #
acq_functions = ["Initial"] # "BayesianUncertainty", "Initial"
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
using Gadfly, Cairo, Fontconfig, DataFrames, CSV
width = 6inch
height = 4inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
Gadfly.push_theme(theme)

# using Plots
# @everywhere using LazyArrays
@everywhere using DistributionsAD

include("./MCMCUtilsRegression.jl")
include("./MCMC_QueryRegression.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

for experiment in experiments
    @everywhere experiment = $experiment
    for (dataset, inout_dims, acquisition_size, n_folds) in zip(datasets, list_inout_dims, acquisition_sizes, list_n_folds)
        # for (dataset, inout_dims, acquisition_size) in zip(datasets, list_inout_dims, acquisition_sizes)
        println(dataset)
        PATH = @__DIR__
        cd(PATH * "/Data/Tabular/Regression/$(dataset)")

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

            pool, test = pool_test_maker(train, test, n_input)
            total_pool_samples = size(pool[1])[2]

            # println("The number of input features are $n_input")
            # println("The number of outputs are $n_output")

            kpi_df = Array{Any}(missing, 0, 15)
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

                                    # n_hidden = n_input > 30 ? 30 : n_input

                                    # nn_initial = Chain(Dense(n_input, n_hidden, relu), Dense(n_hidden, n_hidden, relu), Dense(n_hidden, n_output), softmax)

                                    # # Extract weights and a helper function to reconstruct NN from weights
                                    # parameters_initial, destructured = Flux.destructure(nn_initial)

                                    # feedforward(x, theta) = destructured(theta)(x)

                                    # num_params = length(parameters_initial) # number of paraemters in NN

                                    # network_shape = []
                                    #     (n_hidden, n_input, :relu),
                                    #     (n_hidden, n_hidden, :relu),
                                    #     (n_output, n_hidden, :relu)]

                                    # # Regularization, parameter variance, and total number of
                                    # # parameters.
                                    # num_params = sum([i * o + i for (i, o, _) in network_shape])

                                    include("./BayesianModelMultiProcRegression.jl")

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
                                        prior_std = sqrt.(2 .* vcat(2 / (n_input + l1) * ones(nl1), 2 / (l1 + l2) * ones(nl2), 2 / (l2 + n_output) * ones(n_output_layer)))
                                    end
                                end


                                let
                                    # for acquisition_size in acquisition_sizes
                                    num_mcsteps = 1000

                                    pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(fold)_$(num_chains)_$(num_mcsteps)"
                                    # pipeline_name = "$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(num_chains)_$(num_mcsteps)"
                                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/predictions")
                                    # mkpath("./Experiments/$(experiment)/$(pipeline_name)/hyperpriors")
                                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/classification_performance")
                                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics")
                                    # mkpath("./Experiments/$(experiment)/$(pipeline_name)/independent_param_matrix_all_chains")
                                    # mkpath("./Experiments/$(experiment)/$(pipeline_name)/log_distribution_changes")

                                    n_acq_steps = 1#round(Int, total_pool_samples / acquisition_size, RoundUp)
                                    prior = (zeros(num_params), prior_std)
                                    param_matrix, new_training_data = 0, 0
                                    last_acc = 0
                                    best_acc = 0.5
                                    last_improvement = 0
                                    last_elapsed = 0
                                    benchmark_elapsed = 100.0
                                    new_pool = 0
                                    location_posterior = 0
                                    mcmc_init_params = 0
                                    # if acq_func == "Initial"
                                    for AL_iteration = 1:n_acq_steps
                                        # if last_elapsed >= 2 * benchmark_elapsed
                                        #     @warn(" -> Inference is taking a long time in proportion to Query Size, Increasing Query Size!")
                                        #     acquisition_size = deepcopy(2 * acquisition_size)
                                        #     benchmark_elapsed = deepcopy(last_elapsed)
                                        # end
                                        # if last_acc >= 0.999
                                        #     @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
                                        #     acquisition_size = lastindex(new_pool[2])
                                        # end
                                        # If this is the best accuracy we've seen so far, save the model out
                                        # if last_acc >= best_acc
                                        #     @info(" -> New best accuracy! Logging improvement")
                                        #     best_acc = last_acc
                                        #     last_improvement = AL_iteration
                                        # end
                                        # # If we haven't seen improvement in 5 epochs, drop our learning rate:
                                        # if AL_iteration - last_improvement >= 3 && lastindex(new_pool[2]) > 0
                                        #     @warn(" -> Haven't improved in a while, Increasing Query Size!")
                                        #     n_acq_steps = deepcopy(AL_iteration) - 1
                                        #     break
                                        #     acquisition_size = deepcopy(3 * acquisition_size)
                                        #     benchmark_elapsed = deepcopy(last_elapsed)
                                        #     # After dropping learning rate, give it a few epochs to improve
                                        #     last_improvement = AL_iteration
                                        # end


                                        if AL_iteration == 1
                                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query_regression(prior, pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, "Initial", mcmc_init_params, temperature, prior_informativeness, likelihood_name)
                                            mcmc_init_params = deepcopy(location_posterior)
                                            n_acq_steps = deepcopy(AL_iteration)
                                        elseif lastindex(new_pool[2]) > acquisition_size
                                            if prior_informativeness == "UnInformedPrior"
                                                new_prior = prior
                                            else
                                                new_prior = (location_posterior, prior_std)
                                            end
                                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query_regression(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name)
                                            mcmc_init_params = deepcopy(location_posterior)
                                            n_acq_steps = deepcopy(AL_iteration)
                                        elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                                            if prior_informativeness == "UnInformedPrior"
                                                new_prior = prior
                                            else
                                                new_prior = (location_posterior, prior_std)
                                            end
                                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query_regression(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name)
                                            mcmc_init_params = deepcopy(location_posterior)
                                            println("Trained on last few samples remaining in the Pool")
                                            n_acq_steps = deepcopy(AL_iteration)
                                        end
                                        # num_mcsteps += 500
                                    end

                                    begin
                                        performance_stats = Array{Any}(undef, 5, n_acq_steps)

                                        for al_step = 1:n_acq_steps
                                            data = Array{Any}(undef, 5, num_chains)
                                            for i = 1:num_chains
                                                m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
                                                data[:, i] = m[:, 2]
                                                # rm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv")
                                            end
                                            d = mean(data, dims=2)
                                            writedlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", d)
                                            performance_stats[:, al_step] = d
                                        end

                                        performance_data = Array{Any}(undef, 10, n_acq_steps) #dims=(features, samples(i))
                                        for al_step = 1:n_acq_steps
                                            m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
                                            performance_data[1, al_step] = m[1, 2]#AcquisitionSize
                                            performance_data[2, al_step] = m[2, 2] #MSE
                                            performance_data[3, al_step] = m[3, 2]#MAE

                                            c = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", ',')

                                            performance_data[4, al_step] = acq_func
                                            performance_data[5, al_step] = temperature
                                            performance_data[6, al_step] = experiment

                                            #Cumulative Training Size
                                            if al_step == 1
                                                performance_data[7, al_step] = m[1, 2]
                                            else
                                                performance_data[7, al_step] = performance_data[7, al_step-1] + m[1, 2]
                                            end


                                            performance_data[8, al_step], performance_data[9, al_step], performance_data[10, al_step] = prior_informativeness, prior_variance, likelihood_name

                                        end
                                        kpi = vcat(performance_data, performance_stats)
                                        writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
                                    end
                                    kpi = readdlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", ',')
                                    kpi_df = vcat(kpi_df, permutedims(kpi))
                                end
                            end
                        end
                    end
                end
            end

            kpi_names = vcat([:AcquisitionSize, :MSE, :MAE, :AcquisitionFunction, :Temperature, :Experiment, :CumTrainedSize, :PriorInformativeness, :PriorVariance, :LikelihoodName], :Elapsed, :OOBRhat, :AcceptanceRate, :NumericalErrors, :AvgESS)

            df = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df)
            # CSV.write("./Experiments/$(experiment)/df.csv", df)

            # begin
            #     aucs_acc = []
            #     aucs_t = []
            #     list_compared = []
            #     list_total_training_samples = []
            #     for i in groupby(df, :PriorInformativeness)
            #         acc_ = i.:MSE
            #         time_ = i.:Elapsed
            #         # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
            #         n_aocs_samples = lastindex(acc_)
            #         total_training_samples = i.:CumTrainedSize[end]
            #         push!(list_total_training_samples, total_training_samples)
            #         auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
            #         auc_t = mean(time_[1:n_aocs_samples] .- 0.0) / total_training_samples
            #         push!(list_compared, first(i.:PriorInformativeness))
            #         append!(aucs_acc, (auc_acc))
            #         append!(aucs_t, auc_t)
            #     end
            #     min_total_samples = minimum(list_total_training_samples)
            #     writedlm("./Experiments/$(experiment)/auc_acq_$(fold).txt", [list_compared min_total_samples .* (aucs_acc) min_total_samples .* aucs_t], ',')
            # end

            # df = CSV.read("./Experiments/$(experiment)/df.csv", DataFrame)
            df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            df_folds = vcat(df_folds, df_fold)
        end
        CSV.write("./Experiments/$(experiment)/df_folds.csv", df_folds)

        for (j, i) in enumerate(groupby(df_folds, :LikelihoodName))
            mean_std_MSE = combine(groupby(i, :PriorVariance), :MSE => mean, :MSE => std)
            mean_std_time = combine(groupby(i, :PriorVariance), :Elapsed => mean, :Elapsed => std)
            mean_std_MAE = combine(groupby(i, :PriorVariance), :MAE => mean, :MAE => std)
            mean_std_acceptance_rate = combine(groupby(i, :PriorVariance), :AcceptanceRate => mean, :AcceptanceRate => std)
            mean_std_numerical_errors = combine(groupby(i, :PriorVariance), :NumericalErrors => mean, :NumericalErrors => std)
            group_name = i.LikelihoodName[1]
            CSV.write("./Experiments/$(experiment)/mean_std_MSE$(group_name).csv", mean_std_MSE)
            CSV.write("./Experiments/$(experiment)/mean_std_time$(group_name).csv", mean_std_time)
            CSV.write("./Experiments/$(experiment)/mean_std_MAE$(group_name).csv", mean_std_MAE)
            CSV.write("./Experiments/$(experiment)/mean_std_acceptance_rate$(group_name).csv", mean_std_acceptance_rate)
            CSV.write("./Experiments/$(experiment)/mean_std_numerical_errors$(group_name).csv", mean_std_numerical_errors)
        end


        # df = CSV.read("./Experiments/$(experiment)/df.csv", DataFrame, header=1)

        # df = filter(:AcquisitionFunction => !=("BayesianUncertainty"), df)

        # begin
        #     aucs_acc = []
        #     aucs_t = []
        #     list_compared = []
        #     list_total_training_samples = []
        #     for i in groupby(df, :PriorVariance)
        #         acc_ = i.:MSE
        #         time_ = i.:Elapsed
        #         # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
        #         n_aocs_samples = lastindex(acc_)
        #         total_training_samples = i.:CumTrainedSize[end]
        #         push!(list_total_training_samples, total_training_samples)
        #         auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
        #         auc_t = mean(time_[1:n_aocs_samples] .- 0.0) / total_training_samples
        #         push!(list_compared, first(i.:PriorVariance))
        #         append!(aucs_acc, (auc_acc))
        #         append!(aucs_t, auc_t)
        #     end
        #     min_total_samples = minimum(list_total_training_samples)
        #     writedlm("./Experiments/$(experiment)/auc_acq.txt", [list_compared min_total_samples .* (aucs_acc) min_total_samples .* aucs_t], ',')
        # end

        # # for (j, i) in enumerate(groupby(df, :AcquisitionSize))
        # fig1a = Gadfly.plot(df, x=:CumTrainedSize, y=:MSE, color=:PriorVariance, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.0, ymax=1.0))

        # fig1aa = Gadfly.plot(df, x=:CumTrainedSize, y=:MAE, color=:PriorVariance, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.5, ymax=1.0))

        # fig1b = Gadfly.plot(df, x=:CumTrainedSize, y=:Elapsed, color=:PriorVariance, Geom.point, Geom.line, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=0))

        # fig1a |> PDF("./Experiments/$(experiment)/Accuracy_$(dataset)_$(experiment).pdf", dpi=600)
        # fig1aa |> PDF("./Experiments/$(experiment)/EnsembleMajority_$(dataset)_$(experiment).pdf", dpi=600)
        # fig1b |> PDF("./Experiments/$(experiment)/TrainingTime_$(dataset)_$(experiment).pdf", dpi=600)


        # fig1cc = plot(df, x=:CumTrainedSize, y=:CumCDE, color=:PriorVariance, Geom.point, Geom.line, Guide.ylabel("Cumulative CDE"), Guide.xlabel(nothing), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.0, ymax=1.0))

        # fig1cc |> PDF("./Experiments/$(experiment)/CumCDE_$(dataset)_$(experiment).pdf", dpi=600)

        # # end


        # ### Loops when using Cross Validation for AL
        # begin
        # df_acc = DataFrame()
        # df_time = DataFrame()
        # for prior_name in list_prior_informativeness
        # 	df_acc_ = CSV.read("./Experiments/$(experiment)/mean_std_MSE$(prior_name).csv", DataFrame, header =1)
        # 	df_time_ = CSV.read("./Experiments/$(experiment)/mean_std_time$(prior_name).csv", DataFrame, header =1)

        # 	df_acc_[!, "PriorInformativeness"] .= repeat(prior_name, 5)
        # 	df_time_[!, "PriorInformativeness"] .= repeat(prior_name, 5)

        # 	df_acc = vcat(df_acc, df_acc_)
        # 	df_time = vcat(df_time, df_time_)
        # end

        # fig1a = Gadfly.plot(df_acc, x=:CumTrainedSize, y=:Accuracy_mean, color=:PriorInformativeness, ymin = df_acc.Accuracy_mean - df_acc.Accuracy_std, ymax = df_acc.Accuracy_mean + df_acc.Accuracy_std, Geom.point, Geom.line, Geom.ribbon, yintercept=[0.5], Geom.hline(color=["red"], size=[0.5mm]), Guide.ylabel("MSE"), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=xmin=df_acc.CumTrainedSize[1], ymin=0.0, ymax=1.0))

        # fig1b = Gadfly.plot(df_time, x=:CumTrainedSize, y=:Elapsed_mean, color=:PriorInformativeness, ymin = df_time.Elapsed_mean - df_time.Elapsed_std, ymax = df_time.Elapsed_mean + df_time.Elapsed_std, Geom.point, Geom.line, Geom.ribbon, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=xmin=df_time.CumTrainedSize[1]))

        # fig1a |> PDF("./Experiments/$(experiment)/Accuracy_$(dataset)_$(experiment)_folds.pdf", dpi=600)
        # fig1b |> PDF("./Experiments/$(experiment)/TrainingTime_$(dataset)_$(experiment)_folds.pdf", dpi=600)

        # df = DataFrame()
        # for fold in 1:n_folds
        # 	df_=CSV.read("./Experiments/$(experiment)/auc_acq_$(fold).txt", DataFrame, header=false)
        # 	df = vcat(df, df_)
        # end

        # mean_auc = combine(groupby(df, :Column1), :Column2 => mean, :Column2 => std, :Column3 => mean, :Column3 => std)
        # CSV.write("./Experiments/$(experiment)/mean_auc.txt",mean_auc)
        # end
    end
end