using Distributed
using Turing
num_rounds = 100
datasets = ["secom", "iris", "stroke"]
acq_functions = ["PowerEntropy", "Random"]
acquisition_sizes = [10]
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)

using DataFrames
using CSV
using DelimitedFiles
using XGBoost
using Random
using StatsBase
using Flux: softmax

include("./XGB_Query.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

for dataset in datasets
    experiment_name = "9_xgb_$(dataset)_powerentropy_non_informative_new_batch"

    PATH = @__DIR__
    cd(PATH * "/DataSets/$(dataset)_dataset")


    ###
    ### Data
    ###
    PATH = pwd()
    include(PATH * "/xbg.jl")

    class_names = Array{String}(undef, n_output)
    for i = 1:n_output
        class_names[i] = "$(i)"
    end
    let
        # class_names = Array{String}(undef, n_output)
        # for i = 1:n_output
        # 	class_names[i] = "Class$(i)"
        # end
        kpi_df = Array{Any}(missing, 0, 6 + n_output)
        for acquisition_size in acquisition_sizes
            for acq_func in acq_functions
                pipeline_name = "$(acq_func)_$(acquisition_size)_with_$(num_rounds)_rounds"
                mkpath("./$(experiment_name)/$(pipeline_name)/predictions")
                mkpath("./$(experiment_name)/$(pipeline_name)/classification_performance")
                mkpath("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions")

                AL_iteration = 1
                new_pool, new_xgb, new_training_data = 0, 0, 0
                for AL_iteration = 1:round(Int, total_pool_samples / acquisition_size, RoundUp)
                    if AL_iteration == 1
                        new_pool, new_xgb, new_training_data = xgb_query(new_xgb, pool, new_training_data, input_size, n_output, AL_iteration, test_set, experiment_name, pipeline_name, acquisition_size, num_rounds, "Diversity")

                    elseif lastindex(new_pool[2]) > acquisition_size
                        new_pool, new_xgb, new_training_data = xgb_query(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test_set, experiment_name, pipeline_name, acquisition_size, num_rounds, acq_func)

                    elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                        new_pool, new_xgb, new_training_data = xgb_query(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test_set, experiment_name, pipeline_name, lastindex(new_pool[2]), num_rounds, acq_func)

                        println("Pool exhausted")
                    end
                    AL_iteration += 1
                end

                n_acq_steps = round(Int, total_pool_samples / acquisition_size, RoundUp)

                class_dist_data = Array{Int}(undef, n_output, n_acq_steps)

                performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
                for al_step = 1:n_acq_steps
                    m = readdlm("./$(experiment_name)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
                    performance_data[1, al_step] = m[1, 2]#AcquisitionSize
                    cd = readdlm("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
                    performance_data[2, al_step] = cd[1, 2]#ClassDistEntropy
                    performance_data[3, al_step] = m[2, 2] #Accuracy
                    performance_data[4, al_step] = m[3, 2] #Elapsed

                    for i = 1:n_output
                        class_dist_data[i, al_step] = cd[i+1, 2]
                    end

                end
                kpi = vcat(performance_data, class_dist_data)
                writedlm("./$(experiment_name)/$(pipeline_name)/kpi.csv", kpi, ',')

                # kpi = readdlm("./$(experiment_name)/$(pipeline_name)/kpi.csv", ',')
                # kpi = copy(performance_data)
                cum_acq_train_vector = Array{Int}(undef, 1, n_acq_steps)
                for i = 1:n_acq_steps
                    cum_acq_train_vector[1, i] = sum(kpi[1, 1:i])
                end

                kpi_2 = permutedims(vcat(permutedims(repeat([acq_func], n_acq_steps)), cum_acq_train_vector, kpi))

                # # println(collect(enumerate(cum_acq_train_vector)))
                # kpi_vector = Array{Any}(undef, 1, 6 + n_output)
                # kpi_addition = permutedims(vcat([]))
                # for i=7:6 + n_output
                # 	kpi_vector[1,i] = kpi
                # end

                # for (i, j) in enumerate(cum_acq_train_vector)
                #     kpi_df = vcat(kpi_df, [acq_func acquisition_size j kpi[2, i] kpi[3, i] kpi[4, i]])
                # end

                kpi_df = vcat(kpi_df, kpi_2)
            end
        end
        kpi_names = vcat([:AcquisitionFunction, :CumTrainedSize, :AcquisitionSize, :ClassDistEntropy, :Accuracy, :Elapsed], Symbol.(class_names))
        df = DataFrame(kpi_df, kpi_names)
        CSV.write("./$(experiment_name)/df.csv", df)
    end

    using Gadfly, Cairo, Fontconfig, DataFrames, CSV
    width = 6inch
    height = 20inch
    set_default_plot_size(width, height)

    theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=12pt, key_label_font_size=10pt, key_position=:inside)
    df = CSV.read("./$(experiment_name)/df.csv", DataFrame, header=1)
    Gadfly.push_theme(theme)
    for (j, i) in enumerate(groupby(df, :AcquisitionSize))
        fig1a = plot(i, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples))

        fig1b = plot(i, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples))

        fig1c = plot(i, x=:CumTrainedSize, y=:ClassDistEntropy, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Class Distribution Entropy"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples))

        fig1d = plot(DataFrames.stack(filter(:AcquisitionFunction => ==(acq_functions[1]), i), Symbol.(class_names)), x=:CumTrainedSize, y=:value, color=:variable, Guide.colorkey(title="Class", labels=class_names), Geom.point, Geom.line, Guide.ylabel(acq_functions[1]), Scale.color_discrete_manual("red", "purple", "green"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples, ymin=0, ymax=10))

        fig1e = plot(DataFrames.stack(filter(:AcquisitionFunction => ==(acq_functions[2]), i), Symbol.(class_names)), x=:CumTrainedSize, y=:value, color=:variable, Geom.point, Geom.line, Guide.ylabel(acq_functions[2]), Guide.colorkey(title="Class", labels=class_names), Guide.xlabel("Cumulative Training Size"), Scale.color_discrete_manual("red", "purple", "green"), Coord.cartesian(xmin=0, xmax=total_pool_samples, ymin=0, ymax=10))

        vstack(fig1a, fig1b, fig1c, fig1d, fig1e) |> PNG("./$(experiment_name)/$(experiment_name).png")
    end

end