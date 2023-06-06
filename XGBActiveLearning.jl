using Distributed
using Turing
num_rounds = 200
datasets = ["secom", "iris", "stroke", "banknote", "coalmine", "creditfraud", "creditdefault"]
acq_functions = ["PowerEntropy", "Random"]
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
acquisition_sizes = [20]

    experiment_name = "001_XGB"

    PATH = @__DIR__
    cd(PATH * "/DataSets/$(dataset)_dataset")


    ###
    ### Data
    ###
    PATH = pwd()
	@everywhere PATH = $PATH
    train = CSV.read("train.csv", DataFrame, header=1)
    validate = CSV.read("validate.csv", DataFrame, header=1)
    pool = vcat(train, validate)

    test = CSV.read("test.csv", DataFrame, header=1)


    _, read_dim_cols = size(test)
    n_input = read_dim_cols - 1

    pool, test = pool_test_maker_xgb(pool, test, n_input)
    total_pool_samples = size(pool[1])[2]
	input_size = size(pool[1])[1]
    n_output = lastindex(unique(pool[2]))

    class_names = Array{String}(undef, n_output)
    for i = 1:n_output
        class_names[i] = "$(i-1)"
    end
	
    let
        # class_names = Array{String}(undef, n_output)
        # for i = 1:n_output
        # 	class_names[i] = "Class$(i)"
        # end
        kpi_df = Array{Any}(missing, 0, 7 + n_output)
		for acq_func in acq_functions
			for acquisition_size in acquisition_sizes
                pipeline_name = "$(acq_func)_$(acquisition_size)_with_$(num_rounds)_rounds"
                mkpath("./$(experiment_name)/$(pipeline_name)/predictions")
                mkpath("./$(experiment_name)/$(pipeline_name)/classification_performance")
                mkpath("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions")

				n_acq_steps = round(Int, total_pool_samples / acquisition_size, RoundUp)
                param_matrix, new_training_data = 0, 0
				last_acc = 0
				best_acc = 0.5
    			last_improvement = 0
				last_elapsed = 0
				benchmark_elapsed = 100.0
				new_pool= 0 

                new_pool, new_xgb, new_training_data = 0, 0, 0
                for AL_iteration = 1:n_acq_steps
					if last_elapsed >= 2*benchmark_elapsed
						@warn(" -> Inference is taking a long time in proportion to Query Size, Increasing Query Size!")
						acquisition_size = deepcopy(2*acquisition_size)
						benchmark_elapsed = deepcopy(last_elapsed)
					end
					if last_acc >= 0.999
						@info(" -> Early-exiting: We reached our target accuracy of 99.9%")
						acquisition_size = lastindex(new_pool[2])
					end
					# If this is the best accuracy we've seen so far, save the model out
					if last_acc >= best_acc
						@info(" -> New best accuracy! Logging improvement")
						best_acc = last_acc
						last_improvement = AL_iteration
					end
					 # If we haven't seen improvement in 5 epochs, drop our learning rate:
					 
					if AL_iteration - last_improvement >= 3 && lastindex(new_pool[2]) > 0
						@warn(" -> Haven't improved in a while, Increasing Query Size!")
						acquisition_size = deepcopy(3*acquisition_size)
						benchmark_elapsed = deepcopy(last_elapsed)
						# After dropping learning rate, give it a few epochs to improve
						last_improvement = AL_iteration
					end
                    if AL_iteration == 1
                        new_pool, new_xgb, new_training_data, last_acc, last_elapsed = xgb_query(new_xgb, pool, new_training_data, input_size, n_output, AL_iteration, test, experiment_name, pipeline_name, acquisition_size, num_rounds, acq_func)

                    elseif lastindex(new_pool[2]) > acquisition_size
                        new_pool, new_xgb, new_training_data, last_acc, last_elapsed = xgb_query(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test, experiment_name, pipeline_name, acquisition_size, num_rounds, acq_func)

                    elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                        new_pool, new_xgb, new_training_data, last_acc, last_elapsed = xgb_query(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test, experiment_name, pipeline_name, lastindex(new_pool[2]), num_rounds, acq_func)

                        println("Pool exhausted")
						n_acq_steps = deepcopy(AL_iteration)
                    end
                end


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

				confidences_list = Array{Float32}(undef, 1, n_acq_steps)
                for i = 1:n_acq_steps
                    confidence_avg_ = readdlm("./$(experiment_name)/$(pipeline_name)/predictions/$i.csv", ',')
                    confidence_avg = mean(confidence_avg_)
                    confidences_list[1, i] = confidence_avg
                end

                kpi_2 = permutedims(reduce(vcat, [permutedims(repeat([acq_func], n_acq_steps)), confidences_list, cum_acq_train_vector, kpi]))

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
		kpi_names = vcat([:AcquisitionFunction, :Confidence, :CumTrainedSize, :AcquisitionSize, :ClassDistEntropy, :Accuracy, :Elapsed], Symbol.(class_names))
        df = DataFrame(kpi_df, kpi_names)
        CSV.write("./$(experiment_name)/df.csv", df)
    end

    using Gadfly, Cairo, Fontconfig, DataFrames, CSV
    width = 8inch
    height = 6inch
    set_default_plot_size(width, height)

    theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:right)
    df = CSV.read("./$(experiment_name)/df.csv", DataFrame, header=1)
    Gadfly.push_theme(theme)
    # for (j, i) in enumerate(groupby(df, :AcquisitionSize))
	fig1a = plot(df, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=acquisition_sizes[1], xmax=total_pool_samples, ymin=0.5, ymax=1.0))
	fig1aa = plot(df, x=:CumTrainedSize, y=:Confidence, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]), Guide.xlabel("Cumulative Training Size"), Coord.cartesian(xmin=acquisition_sizes[1], xmax=total_pool_samples, ymin=0.5, ymax=1.0))

	fig1b = plot(df, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Training (seconds)"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples))

	#     fig1c = plot(i, x=:CumTrainedSize, y=:ClassDistEntropy, color=:AcquisitionFunction, Geom.point, Geom.line, Guide.ylabel("Class Distribution Entropy"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples))

	#     fig1d = plot(DataFrames.stack(filter(:AcquisitionFunction => ==(acq_functions[1]), i), Symbol.(class_names)), x=:CumTrainedSize, y=:value, color=:variable, Guide.colorkey(title="Class", labels=class_names), Geom.point, Geom.line, Guide.ylabel(acq_functions[1]), Scale.color_discrete_manual("red", "purple", "green"), Guide.xlabel(nothing), Coord.cartesian(xmin=0, xmax=total_pool_samples, ymin=0, ymax=10))

	#     fig1e = plot(DataFrames.stack(filter(:AcquisitionFunction => ==(acq_functions[2]), i), Symbol.(class_names)), x=:CumTrainedSize, y=:value, color=:variable, Geom.point, Geom.line, Guide.ylabel(acq_functions[2]), Guide.colorkey(title="Class", labels=class_names), Guide.xlabel("Cumulative Training Size"), Scale.color_discrete_manual("red", "purple", "green"), Coord.cartesian(xmin=0, xmax=total_pool_samples, ymin=0, ymax=10))

	#     vstack(fig1a, fig1b, fig1c, fig1d, fig1e) |> PNG("./$(experiment_name)/$(experiment_name).png")
	fig1a |> PNG("./$(experiment_name)/$(experiment_name).png")
	fig1aa |> PNG("./$(experiment_name)/$(experiment_name)_confidence.png")
	fig1b |> PNG("./$(experiment_name)/$(experiment_name)_time.png")
# end

end