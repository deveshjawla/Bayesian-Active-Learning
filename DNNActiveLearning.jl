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

experiments = ["DeepEnsembleWithWiderLayers"]
datasets = ["stroke", "adult1994", "banknote2012"]#20, 20, 10, 10, 20, 20, 10, 40
# acquisition_sizes = [20, 20, 10, 10, 20, 20, 10, 40]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
acquisition_sizes = [40, 120, 40]#80, 120, 40, 40, 80, 150, 50, 494

list_inout_dims = [(4, 2), (4, 2), (4, 2)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_n_folds = [5, 5, 5]#5, 5, 5, 5, 5, 3, 5, 5

acq_functions = ["Initial"] #, "BayesianUncertainty"

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
height = 4inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
Gadfly.push_theme(theme)

include("./DNNUtils.jl")
include("./DNN_Query.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

for experiment in experiments
    @everywhere experiment = $experiment

    for (dataset, inout_dims, acquisition_size, n_folds) in zip(datasets, list_inout_dims, acquisition_sizes, list_n_folds)
        println(dataset)
        PATH = @__DIR__
        cd(PATH * "/Data/Tabular/$(dataset)")

        n_input, n_output = inout_dims

		@everywhere begin
			n_input = $n_input
			n_output = $n_output
			using Flux        ###
			### Dense Network specifications(Functional Model)
			###
			# include(PATH * "/Network.jl")
			using Statistics: mean
			function logitcrossentropyweighted(ŷ::AbstractArray, y::AbstractArray, sample_weights::AbstractArray; dims=1)
				if size(ŷ) != size(y)
					error("logitcrossentropyweighted(ŷ, y), sizes of (ŷ, y) are not the same")
				end
				mean(.-sum(sample_weights .* (y .* logsoftmax(ŷ; dims=dims)); dims=dims))
			end
		end

        ###
        ### Data
        ###

        PATH = pwd()
        @everywhere PATH = $PATH

        df_folds = DataFrame()
        for fold in 1:n_folds
            train = CSV.read("./FiveFolds/train_$(fold).csv", DataFrame, header=1)
            # validate = CSV.read("validate.csv", DataFrame, header=1)
            # pool = vcat(train, validate)

            test = CSV.read("./FiveFolds/test_$(fold).csv", DataFrame, header=1)

            pool, test = pool_test_maker(train, test, n_input)
            total_pool_samples = size(pool[1])[2]

            class_names = Array{String}(undef, n_output)
            for i = 1:n_output
                class_names[i] = "$(i)"
            end

            kpi_df = Array{Any}(missing, 0, 9 + 2 * n_output)
            for acq_func in acq_functions
                @everywhere acq_func = $acq_func

                let
                    num_mcsteps = 20

                    pipeline_name = "$(acquisition_size)_$(acq_func)_$(fold)"
                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/predictions")
                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/classification_performance")
                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics")
                    # mkpath("./Experiments/$(experiment)/$(pipeline_name)/independent_param_matrix_all_chains")
                    # mkpath("./Experiments/$(experiment)/$(pipeline_name)/log_distribution_changes")
                    mkpath("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions")

                    n_acq_steps = 10#round(Int, total_pool_samples / acquisition_size, RoundUp)
                    param_matrix, new_training_data = 0, 0
                    last_acc = 0
                    best_acc = 0.5
                    last_improvement = 0
                    last_elapsed = 0
                    benchmark_elapsed = 100.0
                    new_pool = 0
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
                        # # If this is the best accuracy we've seen so far, save the model out
                        # if last_acc >= best_acc
                        #     @info(" -> New best accuracy! Logging improvement")
                        #     best_acc = last_acc
                        #     last_improvement = AL_iteration
                        # end
                        # # If we haven't seen improvement in 5 epochs, drop our learning rate:

                        # if AL_iteration - last_improvement >= 3 && lastindex(new_pool[2]) > 0
                        #     @warn(" -> Haven't improved in a while, Increasing Query Size!")
                        #     acquisition_size = deepcopy(3 * acquisition_size)
                        #     benchmark_elapsed = deepcopy(last_elapsed)
                        #     # After dropping learning rate, give it a few epochs to improve
                        #     last_improvement = AL_iteration
                        # end
                        if AL_iteration == 1
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, "Initial")
                            n_acq_steps = deepcopy(AL_iteration)
                        elseif lastindex(new_pool[2]) > acquisition_size
                            # new_prior = (new_prior[1], sigma)
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, acq_func)
                            n_acq_steps = deepcopy(AL_iteration)
                        elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                            # new_prior = (new_prior[1], sigma)
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), num_mcsteps, num_chains, acq_func)
                            println("Trained on last few samples remaining in the Pool")
                            n_acq_steps = deepcopy(AL_iteration)
                        end
                        # num_mcsteps += 500
                    end

                    class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
                    cum_class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
                    performance_data = Array{Any}(undef, 8, n_acq_steps) #dims=(features, samples(i))
                    cum_class_dist_ent = Array{Any}(undef, 1, n_acq_steps)
                    for al_step = 1:n_acq_steps
                        m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
                        performance_data[1, al_step] = m[1, 2]#AcquisitionSize
                        cd = readdlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
                        performance_data[2, al_step] = cd[1, 2]#ClassDistEntropy
                        performance_data[3, al_step] = m[2, 2] #Accuracy

                        ensemble_majority_avg_ = readdlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv", ',')
                        ensemble_majority_avg = mean(ensemble_majority_avg_[2, :])
                        performance_data[4, al_step] = ensemble_majority_avg
                        rm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv")

                        c = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step).csv", ',')
                        performance_data[5, al_step] = c[1] #Elapsed

                        performance_data[6, al_step] = acq_func
                        performance_data[7, al_step] = experiment

                        #Cumulative Training Size
                        if al_step == 1
                            performance_data[8, al_step] = m[1, 2]
                        else
                            performance_data[8, al_step] = performance_data[8, al_step-1] + m[1, 2]
                        end

                        for i = 1:n_output
                            class_dist_data[i, al_step] = cd[i+1, 2]
                            if al_step == 1
                                cum_class_dist_data[i, al_step] = cd[i+1, 2]
                            elseif al_step > 1
                                cum_class_dist_data[i, al_step] = cum_class_dist_data[i, al_step-1] + cd[i+1, 2]
                            end
                        end
                        cum_class_dist_ent[1, al_step] = normalized_entropy(softmax(cum_class_dist_data[:, al_step]), n_output)
                    end
                    kpi = vcat(performance_data, class_dist_data, cum_class_dist_data, cum_class_dist_ent)
                    writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
                    kpi = readdlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", ',')
                    kpi_df = vcat(kpi_df, permutedims(kpi))

                end
            end
            kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :Accuracy, :EnsembleMajority, :Elapsed, :AcquisitionFunction, :Experiment, :CumTrainedSize], Symbol.(class_names), Symbol.(class_names), :CumCDE)
            df = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df)

            # df = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            df_folds = vcat(df_folds, df)
        end

        for (j, i) in enumerate(groupby(df_folds, :AcquisitionFunction))
            mean_std_acc = combine(groupby(i, :AcquisitionFunction), :Accuracy => mean, :Accuracy => std)
            mean_std_time = combine(groupby(i, :AcquisitionFunction), :Elapsed => mean, :Elapsed => std)
            mean_std_ensemble_majority = combine(groupby(i, :AcquisitionFunction), :EnsembleMajority => mean, :EnsembleMajority => std)
            acquisition_function = i.AcquisitionFunction[1]
            CSV.write("./Experiments/$(experiment)/mean_std_acc$(acquisition_function).csv", mean_std_acc)
            CSV.write("./Experiments/$(experiment)/mean_std_time$(acquisition_function).csv", mean_std_time)
            CSV.write("./Experiments/$(experiment)/mean_std_ensemble_majority$(acquisition_function).csv", mean_std_ensemble_majority)
        end


        # sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
        # _, i = findmax(chain[:lp])
        # i = i.I[1]
        # elapsed = chain_timed.time
        # θ = MCMCChains.group(chain, :θ).value
        # θ[i, :]

        # end


        # # df = CSV.read("./Experiments/$(experiment)/df.csv", DataFrame, header=1)

        # # df = filter(:AcquisitionFunction => !=("BayesianUncertainty"), df)

        # begin
        #     aucs_acc = []
        #     aucs_t = []
        #     list_compared = []
        #     list_total_training_samples = []
        #     for i in groupby(df, :AcquisitionFunction)
        #         acc_ = i.:Accuracy
        #         time_ = i.:Elapsed
        #         # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
        #         n_aocs_samples = lastindex(acc_)
        #         total_training_samples = i.:CumTrainedSize[end]
        #         push!(list_total_training_samples, total_training_samples)
        #         auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
        #         auc_t = mean(time_[1:n_aocs_samples] .- 0.0) / total_training_samples
        #         push!(list_compared, first(i.:AcquisitionFunction))
        #         append!(aucs_acc, (auc_acc))
        #         append!(aucs_t, auc_t)
        #     end
        #     min_total_samples = minimum(list_total_training_samples)
        #     writedlm("./Experiments/$(experiment)/auc_acq.txt", [list_compared min_total_samples .* (aucs_acc) min_total_samples .* aucs_t], ',')
        # end

        # # for (j, i) in enumerate(groupby(df, :AcquisitionSize))
        # fig1a = Gadfly.plot(df, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.0, ymax=1.0))

        # fig1aa = Gadfly.plot(df, x=:CumTrainedSize, y=:EnsembleMajority, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.5, ymax=1.0))

        # fig1b = Gadfly.plot(df, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=0))

        # fig1a |> PDF("./Experiments/$(experiment)/Accuracy_$(dataset)_$(experiment).pdf", dpi=600)
        # fig1aa |> PDF("./Experiments/$(experiment)/EnsembleMajority_$(dataset)_$(experiment).pdf", dpi=600)
        # fig1b |> PDF("./Experiments/$(experiment)/TrainingTime_$(dataset)_$(experiment).pdf", dpi=600)

        # fig1c = plot(df, x=:CumTrainedSize, y=:ClassDistEntropy, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Class Distribution Entropy"), Guide.xlabel(nothing), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.0, ymax=1.0))

        # fig1cc = plot(df, x=:CumTrainedSize, y=:CumCDE, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Cumulative CDE"), Guide.xlabel(nothing), Coord.cartesian(xmin=df.AcquisitionSize[1], ymin=0.0, ymax=1.0))

        # fig1c |> PDF("./Experiments/$(experiment)/ClassDistEntropy_$(dataset)_$(experiment).pdf", dpi=600)
        # fig1cc |> PDF("./Experiments/$(experiment)/CumCDE_$(dataset)_$(experiment).pdf", dpi=600)

        #     fig1d = plot(DataFrames.stack(filter(:Experiment => ==(acq_functions[1]), i), Symbol.(class_names)), x=:CumTrainedSize, y=:value, color=:variable, Guide.colorkey(title="Class", labels=class_names), Geom.point, Geom.line, Guide.ylabel(acq_functions[1]), Scale.color_discrete_manual("red", "purple", "green"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples, ymin=0, ymax=10))
        # end

    end
end