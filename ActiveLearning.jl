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
num_mcsteps = 100
datasets = ["secom", "iris", "stroke"]
acq_functions = ["PowerEntropy", "Random"]
acquisition_sizes = [10]
# Add four processes to use for sampling.
addprocs(num_chains; exeflags=`--project`)

using DataFrames
using CSV
using DelimitedFiles
using Random
using StatsBase
using Distances

include("./BNNUtils.jl")
include("./BNN_Query.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

for dataset in datasets
    experiment_name = "9_TDIST_$(dataset)_powerentropy_non_informative_cumulative"
    PATH = @__DIR__
    cd(PATH * "/DataSets/$(dataset)_dataset")

    ###
    ### Data
    ###

    PATH = pwd()
    include(PATH * "/Data.jl")
    @everywhere PATH = $PATH

    ###
    ### Dense Network specifications(Functional Model)
    ###

    input_size = size(pool[1])[1]
    n_output = lastindex(unique(pool[2]))
    # println("The number of input features are $input_size")
    # println("The number of outputs are $n_output")
    class_names = Array{String}(undef, n_output)
    for i = 1:n_output
        class_names[i] = "$(i)"
    end

    @everywhere begin
        input_size = $input_size
        n_output = $n_output
        using Flux, Turing

        include(PATH * "/Network.jl")

        # setprogress!(false)
        # using Zygote
        # Turing.setadbackend(:zygote)
        using ReverseDiff
        Turing.setadbackend(:reversediff)

        #Here we define the layer by layer initialisation
        sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
        # sigma = ones(total_num_params)

    end

    include("./BayesianModelMultiProc.jl")

    let
        kpi_df = Array{Any}(missing, 0, 6 + n_output)
        for acquisition_size in acquisition_sizes
            for acq_func in acq_functions
                pipeline_name = "$(acq_func)_$(acquisition_size)_with_$(num_mcsteps)_MCsteps"
                mkpath("./$(experiment_name)/$(pipeline_name)/predictions")
                mkpath("./$(experiment_name)/$(pipeline_name)/classification_performance")
                mkpath("./$(experiment_name)/$(pipeline_name)/convergence_statistics")
                mkpath("./$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains")
                mkpath("./$(experiment_name)/$(pipeline_name)/log_distribution_changes")
                mkpath("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions")


                AL_iteration = 1
                prior = 0
                new_pool, new_prior, param_matrix, new_training_data = 0, 0, 0, 0
                for AL_iteration = 1:round(Int, total_pool_samples / acquisition_size, RoundUp)
                    if AL_iteration == 1
                        location_prior, scale_prior = zeros(total_num_params), sigma
                        # @everywhere location_prior = $location_prior
                        # @everywhere scale_prior = $scale_prior
                        prior = (location_prior, scale_prior)
                        # @everywhere prior = $prior
                        new_pool, new_prior, param_matrix, new_training_data = bnn_query(prior, pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, experiment_name, pipeline_name, acquisition_size, num_mcsteps, num_chains, "Diversity")
                    elseif lastindex(new_pool[2]) > acquisition_size
                        new_prior = (new_prior[1], sigma)
                        new_pool, new_prior, param_matrix, new_training_data = bnn_query(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, experiment_name, pipeline_name, acquisition_size, num_mcsteps, num_chains, acq_func)
                    elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
                        new_prior = (new_prior[1], sigma)
                        new_pool, new_prior, param_matrix, new_training_data = bnn_query(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, experiment_name, pipeline_name, lastindex(new_pool[2]), num_mcsteps, num_chains, acq_func)
                        println("Pool exhausted")
                    end
                    AL_iteration += 1
                end

                n_acq_steps = round(Int, total_pool_samples / acquisition_size, RoundUp)

                class_dist_data = Array{Int}(undef, n_output, n_acq_steps)

                for al_step = 1:n_acq_steps
                    data = Array{Any}(undef, 5, num_chains)
                    for i = 1:num_chains
                        m = readdlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
                        data[:, i] = m[:, 2]
                        rm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv")
                    end
                    d = mean(data, dims=2)
                    writedlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", d)
                end

                performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
                for al_step = 1:n_acq_steps
                    m = readdlm("./$(experiment_name)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
                    performance_data[1, al_step] = m[1, 2]#AcquisitionSize
                    cd = readdlm("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
                    performance_data[2, al_step] = cd[1, 2]#ClassDistEntropy
                    performance_data[3, al_step] = m[2, 2] #Accuracy
                    c = readdlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", ',')
                    performance_data[4, al_step] = c[1] #Elapsed

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
                # println(collect(enumerate(temporary_vector)))
                # for (i, j) in enumerate(temporary_vector)
                # 	kpi_df = vcat(kpi_df, [acq_func acquisition_size j kpi[2, i] kpi[3, i] kpi[4, i]])
                # end
                kpi_df = vcat(kpi_df, kpi_2)
            end
        end
        kpi_names = vcat([:AcquisitionFunction, :CumTrainedSize, :AcquisitionSize, :ClassDistEntropy, :Accuracy, :Elapsed], Symbol.(class_names))
        df = DataFrame(kpi_df, kpi_names)
        CSV.write("./$(experiment_name)/df.csv", df)
    end

    # sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
    # _, i = findmax(chain[:lp])
    # i = i.I[1]
    # elapsed = chain_timed.time
    # θ = MCMCChains.group(chain, :θ).value
    # θ[i, :]

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