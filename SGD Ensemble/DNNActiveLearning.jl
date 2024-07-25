using Distributed
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)

experiments = ["IncrementalLearningDNN"]
datasets = ["iris1988", "yeast1996"]#20, 20, 10, 10, 20, 20, 10, 40
# acquisition_sizes = [20, 20, 10, 10, 20, 20, 10, 40]#"stroke", "adult1994", "banknote2012", "creditfraud", "creditdefault2005", "coalmineseismicbumps",  "iris1988", "yeast1996"
minimum_training_sizes = [30, 296] #60, 60, 40, 40, 80, 100, 30, 296
acquisition_sizes = round.(Int, minimum_training_sizes ./ 10)
list_acq_steps = [10, 10]

list_inout_dims = [(4, 3), (8, 10)] # (4, 2), (4, 2), (4, 2), (28, 2), (22, 2), (11, 2), (4, 3), (8, 10)

list_n_folds = [5, 5]#5, 5, 5, 5, 5, 3, 5, 5
acq_functions = ["Random"] #, "BayesianUncertainty"

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
width = 6inch
height = 4inch
set_default_plot_size(width, height)
theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
Gadfly.push_theme(theme)

include("./DNNUtils.jl")
include("./DNN_Query.jl")
include("./../DataUtils.jl")
include("./../ScoringFunctions.jl")
include("./../AcquisitionFunctions.jl")

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
            using Flux###
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

            test = CSV.read("./FiveFolds/test_$(fold).csv", DataFrame, header=1)

            pool, test = pool_test_to_matrix(train, test, n_input, "MCMC")
            total_pool_samples = size(pool[1])[2]

            class_names = Array{String}(undef, n_output)
            for i = 1:n_output
                class_names[i] = "$(i)"
            end

            kpi_df = Array{Any}(missing, 0, 10 + 2 * n_output)
            for acq_func in acq_functions
                @everywhere acq_func = $acq_func

                let
                    ensemble_size = 100

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
                        if AL_iteration == 1
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, ensemble_size, "Initial")
                            n_acq_steps = deepcopy(AL_iteration)
                        elseif lastindex(new_pool[2]) > acquisition_size
                            # new_prior = (new_prior[1], sigma)
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, acquisition_size, ensemble_size, acq_func)
                            n_acq_steps = deepcopy(AL_iteration)
                        elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                            # new_prior = (new_prior[1], sigma)
                            new_pool, param_matrix, new_training_data, last_acc, last_elapsed = dnn_query(new_pool, new_training_data, n_input, n_output, param_matrix, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), ensemble_size, acq_func)
                            println("Trained on last few samples remaining in the Pool")
                            n_acq_steps = deepcopy(AL_iteration)
                        end
                    end

                    class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
                    cum_class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
                    performance_data = Array{Any}(undef, 9, n_acq_steps) #dims=(features, samples(i))
                    cum_class_dist_ent = Array{Any}(undef, 1, n_acq_steps)
                    for al_step = 1:n_acq_steps
                        m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
                        performance_data[1, al_step] = m[1, 2]#AcquisitionSize
                        cd = readdlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
                        performance_data[2, al_step] = cd[1, 2]#ClassDistEntropy
                        performance_data[3, al_step] = m[2, 2] #Accuracy Score
                        performance_data[9, al_step] = m[3, 2]#F1 

                        ensemble_majority_avg_ = readdlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv", ',')
                        ensemble_majority_avg = mean(ensemble_majority_avg_[2, :])
                        performance_data[4, al_step] = ensemble_majority_avg
                        # rm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv")

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
            kpi_names = vcat([:AcquisitionSize, :ClassDistEntropy, :WeightedAccuracy, :EnsembleMajority, :Elapsed, :AcquisitionFunction, :Experiment, :CumTrainedSize, :WeightedF1], Symbol.(class_names), Symbol.(class_names), :CumCDE)
            df = DataFrame(kpi_df, kpi_names; makeunique=true)
            CSV.write("./Experiments/$(experiment)/df_$(fold).csv", df)

            begin
                aucs_acc = []
                aucs_t = []
                list_compared = []
                list_total_training_samples = []
                for i in groupby(df, :AcquisitionFunction)
                    acc_ = i.:WeightedAccuracy
                    time_ = i.:Elapsed
                    # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
                    n_aocs_samples = lastindex(acc_)
                    total_training_samples = i.:CumTrainedSize[end]
                    push!(list_total_training_samples, total_training_samples)
                    auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
                    auc_t = mean(time_[1:n_aocs_samples] .- 0.0) / total_training_samples
                    push!(list_compared, first(i.:AcquisitionFunction))
                    append!(aucs_acc, (auc_acc))
                    append!(aucs_t, auc_t)
                end
                min_total_samples = minimum(list_total_training_samples)
                writedlm("./Experiments/$(experiment)/auc_$(fold).csv", [list_compared min_total_samples .* (aucs_acc) min_total_samples .* aucs_t], ',')
            end

            # df = CSV.read("./Experiments/$(experiment)/df.csv", DataFrame)
            df_fold = CSV.read("./Experiments/$(experiment)/df_$(fold).csv", DataFrame)
            df_folds = vcat(df_folds, df_fold)
        end
        CSV.write("./Experiments/$(experiment)/df_folds.csv", df_folds)

        for (j, i) in enumerate(groupby(df_folds, :AcquisitionFunction))
            mean_std_acc = combine(groupby(i, :CumTrainedSize), :WeightedAccuracy => mean, :WeightedAccuracy => std)
            mean_std_f1 = combine(groupby(i, :CumTrainedSize), :WeightedF1 => mean, :WeightedF1 => std)
            mean_std_time = combine(groupby(i, :CumTrainedSize), :Elapsed => mean, :Elapsed => std)
            mean_std_ensemble_majority = combine(groupby(i, :CumTrainedSize), :EnsembleMajority => mean, :EnsembleMajority => std)
            acquisition_function = i.AcquisitionFunction[1]
            CSV.write("./Experiments/$(experiment)/mean_std_acc$(acquisition_function).csv", mean_std_acc)
            CSV.write("./Experiments/$(experiment)/mean_std_f1$(acquisition_function).csv", mean_std_f1)
            CSV.write("./Experiments/$(experiment)/mean_std_time$(acquisition_function).csv", mean_std_time)
            CSV.write("./Experiments/$(experiment)/mean_std_ensemble_majority$(acquisition_function).csv", mean_std_ensemble_majority)
        end

        ### Loops when using Cross Validation for AL
        begin
            df_acc = DataFrame()
            df_f1 = DataFrame()
            df_time = DataFrame()
            for acq_func in acq_functions
                df_acc_ = CSV.read("./Experiments/$(experiment)/mean_std_acc$(acq_func).csv", DataFrame, header=1)
                df_f1_ = CSV.read("./Experiments/$(experiment)/mean_std_f1$(acq_func).csv", DataFrame, header=1)
                df_time_ = CSV.read("./Experiments/$(experiment)/mean_std_time$(acq_func).csv", DataFrame, header=1)

                df_acc_[!, "AcquisitionFunction"] .= repeat(acq_func, 5)
                df_f1_[!, "AcquisitionFunction"] .= repeat(acq_func, 5)
                df_time_[!, "AcquisitionFunction"] .= repeat(acq_func, 5)

                df_acc = vcat(df_acc, df_acc_)
                df_f1 = vcat(df_f1, df_f1_)
                df_time = vcat(df_time, df_time_)
            end

            fig1a = Gadfly.plot(df_acc, x=:CumTrainedSize, y=:WeightedAccuracy_mean, color=:AcquisitionFunction, ymin=df_acc.Accuracy_mean - df_acc.Accuracy_std, ymax=df_acc.Accuracy_mean + df_acc.Accuracy_std, Geom.point, Geom.line, Geom.ribbon, yintercept=[0.5], Geom.hline(color=["red"], size=[0.5mm]), Guide.ylabel("Accuracy"), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df_acc.CumTrainedSize[1], ymin=0.0, ymax=1.0))
            fig1aa = Gadfly.plot(df_f1, x=:CumTrainedSize, y=:WeightedF1_mean, color=:AcquisitionFunction, ymin=df_f1.F1_mean - df_f1.F1_std, ymax=df_f1.F1_mean + df_f1.F1_std, Geom.point, Geom.line, Geom.ribbon, yintercept=[0.5], Geom.hline(color=["red"], size=[0.5mm]), Guide.ylabel("F1"), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=df_f1.CumTrainedSize[1], ymin=0.0, ymax=1.0))

            fig1b = Gadfly.plot(df_time, x=:CumTrainedSize, y=:Elapsed_mean, color=:AcquisitionFunction, ymin=df_time.Elapsed_mean - df_time.Elapsed_std, ymax=df_time.Elapsed_mean + df_time.Elapsed_std, Geom.point, Geom.line, Geom.ribbon, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=df_time.CumTrainedSize[1]))

            fig1a |> PDF("./Experiments/$(experiment)/Accuracy_$(dataset)_$(experiment)_folds.pdf", dpi=600)
            fig1aa |> PDF("./Experiments/$(experiment)/F1_$(dataset)_$(experiment)_folds.pdf", dpi=600)
            fig1b |> PDF("./Experiments/$(experiment)/TrainingTime_$(dataset)_$(experiment)_folds.pdf", dpi=600)

            df = DataFrame()
            for fold in 1:n_folds
                df_ = CSV.read("./Experiments/$(experiment)/auc_$(fold).csv", DataFrame, header=false)
                df = vcat(df, df_)
            end

            mean_auc = combine(groupby(df, :Column1), :Column2 => mean, :Column2 => std, :Column3 => mean, :Column3 => std)
            CSV.write("./Experiments/$(experiment)/mean_auc.csv", mean_auc, header=[:AcquisitionFunction, :AccAUC_mean, :AccAUC_std, :TimeAUC_mean, :TimeAUC_std])
        end
    end
end