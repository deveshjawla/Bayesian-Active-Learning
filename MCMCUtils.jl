
using Flux: activations
function activations_weighted_sums(test_xs::Array{Float32,1}, params_set::Array{Float32,2}, directory_plots, n_layers)
    nets = map(feedforward, eachrow(params_set))
    activations_weights = map(x -> activations_weights_df_aggregator(x, test_xs, n_layers), nets)
    activations_df = DataFrame()
    for i = 1:lastindex(nets)
        activations_df = vcat(activations_df, activations_weights[i][1])
    end
    weights_df = DataFrame()
    for i = 1:lastindex(nets)
        weights_df = vcat(weights_df, activations_weights[i][2])
    end
    activations_weights_plot = plot_maker_activations_weights(activations_df, weights_df)
    activations_weights_plot |> PDF("$(directory_plots)/activations_weights.pdf", dpi=300)
    return nothing
end

function activations_weights_df_aggregator(ch, input, n_layers)
    _activations = activations(ch, input)
    # last_layer = lastindex(_activations)
    # n_layers = lastindex(_activations)
    # if n_layers != last_layer
    # 	error("Check number of layers")
    # end
    _params = Flux.params(ch)
    df_activations = DataFrame()
    df_weights = DataFrame()
    for layer in 1:n_layers
        acts_df = _activations[layer]
        if layer == n_layers #last_layer
            acts_df = tanh.(_activations[layer])
        end
        layers_df = [layer for x in 1:lastindex(acts_df)]
        df_activations = vcat(df_activations, DataFrame(Activations=acts_df, Layer=layers_df))
        params_df = vec(_params[layer*2-1])
        layers_df = [layer for x in 1:lastindex(params_df)]
        df_weights = vcat(df_weights, DataFrame(Weights=params_df, Layer=layers_df))
    end
    return df_activations, df_weights
end

function plot_maker_activations_weights(df_activations, df_weights)
    width = 6inch
    height = 8inch
    set_default_plot_size(width, height)
    theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
    Gadfly.push_theme(theme)
    myviolinplot_activations = plot(df_activations, x=:Layer, y=:Activations, Geom.violin)
    myboxplot_activations = plot(df_activations, x=:Layer, y=:Activations, Geom.boxplot)
    myviolinplot = plot(df_weights, x=:Layer, y=:Weights, Geom.violin)
    myboxplot = plot(df_weights, x=:Layer, y=:Weights, Geom.boxplot)
    activations_weights = gridstack([myviolinplot_activations myboxplot_activations; myviolinplot myboxplot])
    return activations_weights
end

function convergence_stats(i::Int, chains, elapsed::Float32)::Tuple{Float32,Int64,Float32,Float32,Float32,Float32}
    gelman = gelmandiag(chains)
    psrf_psrfci = convert(Array, gelman)
    max_psrf = maximum(psrf_psrfci[:, 1])

    ch = chains[:, :, i]
    summaries, quantiles = describe(ch)
    large_rhat = count(>=(1.01), sort(summaries[:, :rhat]))
    small_rhat = count(<=(0.99), sort(summaries[:, :rhat]))
    oob_rhat = large_rhat + small_rhat
    avg_acceptance_rate = mean(ch[:acceptance_rate])
    total_numerical_error = sum(ch[:numerical_error])
    avg_ess = mean(summaries[:, :ess_per_sec])
    # println(describe(summaries[:, :mean]))
    return elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf
end

"""
### It analyses predictions of an ensemble for a classification task.

	The Ensemble might have noise_x explicitely modeled into it. For models using reconstruct, the reconstruct of the model must be provided.

	param_matrix = is the matrix of model parameters of Dims(Number of Models, Number of Parameters in each model)

	test_xs = the matrix of test dataset with Dims(Number of Input features, Number of Samples)


	Returns: Matrix with Dims(6, Number of Samples) where 6 are the following quantitites = [pred_label, pred_prob, pred_std, aleatoric_uncertainties, epistemic_uncertainties, total_uncertainties]
"""
function pred_analyzer_multiclass(test_xs::Array{Float32,2}, param_matrix::Array{Float32,2}; noise_set_x=nothing, noise_set_y=nothing, reconstruct=nothing, output_activation_function="Softmax")::Array{Float32,2}
    if isnothing(reconstruct)
        nets = map(feedforward, eachrow(param_matrix))
    else
        nets = map(reconstruct, eachrow(param_matrix))
    end

    if output_activation_function == "Softmax"
        if isnothing(noise_set_x) && isnothing(noise_set_y)
            # predictions_nets = map(x -> softmax(x(test_xs)), nets)
            predictions_nets = map(net -> softmax(net(test_xs); dims=1), nets)
        elseif isnothing(noise_set_y)
            predictions_nets = map((net, noise_x) -> softmax(net(test_xs .+ noise_x); dims=1), nets, noise_set_x)
        elseif isnothing(noise_set_x)
            predictions_nets = map((net, noise_y) -> softmax(net(test_xs) .+ noise_y; dims=1), nets, noise_set_y)
        else
            predictions_nets = map((net, noise_x, noise_y) -> softmax(net(test_xs .+ noise_x) .+ noise_y; dims=1), nets, noise_set_x, noise_set_y)
        end
    elseif output_activation_function == "Relu"
        if isnothing(noise_set_x) && isnothing(noise_set_y)
            # predictions_nets = map(x -> softmax(x(test_xs)), nets)
            predictions_nets = map(net -> net(test_xs) ./ sum(net(test_xs), dims=1), nets)
        elseif isnothing(noise_set_y)
            predictions_nets = map((net, noise_x) -> net(test_xs .+ noise_x) ./ sum(net(test_xs .+ noise_x), dims=1), nets, noise_set_x)
        elseif isnothing(noise_set_x)
            predictions_nets = map((net, noise_y) -> (net(test_xs) .+ noise_y) ./ sum(net(test_xs) .+ noise_y, dims=1), nets, noise_set_y)
        else
            predictions_nets = map((net, noise_x, noise_y) -> (net(test_xs .+ noise_x) .+ noise_y) ./ sum((net(test_xs .+ noise_x) .+ noise_y), dims=1), nets, noise_set_x, noise_set_y)
        end
    end

    # Determine the size of the final 3D array
    num_matrices = lastindex(predictions_nets)
    rows, cols = size(predictions_nets[1])

    # Preallocate the 3D array
    pred_matrix = Array{Float32}(undef, rows, cols, num_matrices)

    # Fill the preallocated array
    for i in 1:num_matrices
        pred_matrix[:, :, i] = predictions_nets[i]
    end

    mean_pred_matrix = mapslices(x -> mean(x, dims=2), pred_matrix, dims=[1, 3])
    std_pred_matrix = mapslices(x -> std(x, dims=2), pred_matrix, dims=[1, 3])
    bald_scores = mapslices(x -> bald(x, first(size(x))), pred_matrix, dims=[1, 3])
    aleatoric_uncertainties = mapreduce(x -> x[2], hcat, bald_scores[1, :, 1])
    epistemic_uncertainties = mapreduce(x -> x[3], hcat, bald_scores[1, :, 1])
    total_uncertainties = mapreduce(x -> x[1], hcat, bald_scores[1, :, 1])

    # ensembles = mapreduce(x -> mapslices(argmax, x, dims=1), vcat, predictions_nets)
    # pred_plus_std = mapslices(majority_voting, ensembles, dims=1)

    pred_label = map(argmax, eachcol(mean_pred_matrix[:, :, 1]))
    pred_prob = map(maximum, eachcol(mean_pred_matrix[:, :, 1]))
    pred_std = [std_pred_matrix[x, i, 1] for (i, x) in enumerate(pred_label)]
    pred_plus_std = vcat(permutedims(pred_label), permutedims(pred_prob), permutedims(pred_std))

    pred_matrix = vcat(pred_plus_std, aleatoric_uncertainties, epistemic_uncertainties, total_uncertainties)
    return pred_matrix#[[1, 4], :]#3,4,5
end

"""
Returns a tuple of {Prediction, Prediction probability}

Uses a simple argmax and percentage of samples in the ensemble respectively
"""
function pred_analyzer_regression(test_xs::Array{Float32,2}, param_matrix::Array{Float32,2}; reconstruct=nothing)::Tuple{Array{Float32,2},Array{Float32,2}}
    if isnothing(reconstruct)
        nets = map(feedforward, eachrow(param_matrix))
    else
        nets = map(reconstruct, eachrow(param_matrix))
    end
    predictions_nets = map(net -> net(test_xs), nets)
    predictions = permutedims(reduce(vcat, predictions_nets))
    pred_matrix_mean = mapslices(mean, predictions, dims=2)
    pred_matrix_std = mapslices(std, predictions, dims=2)
    return pred_matrix_mean, pred_matrix_std
end


function collecting_stats_active_learning_experiments_regression(n_acq_steps, experiment, pipeline_name, num_chains, learning_algorithm)
    performance_stats = Array{Any}(undef, 5, n_acq_steps)
    for al_step = 1:n_acq_steps
        if learning_algorithm == "MCMC"
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
    if learning_algorithm == "MCMC"
        kpi = vcat(performance_data, performance_stats)
    elseif learning_algorithm == "VI"
        kpi = performance_data
    end
    writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
    return kpi
end

function collecting_stats_active_learning_experiments_classification(n_acq_steps, experiment, pipeline_name, num_chains, n_output, learning_algorithm)
    performance_stats = Array{Float32}(undef, 6, n_acq_steps)
    elapsed_stats = Array{Float32}(undef, 1, n_acq_steps)
    class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    cum_class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    performance_data = Array{Any}(undef, 11, n_acq_steps) #dims=(features, samples(i))
    cum_class_dist_ent = Array{Float32}(undef, 1, n_acq_steps)

    for al_step = 1:n_acq_steps
        if learning_algorithm == "MCMC"
            data = Array{Any}(undef, 6, num_chains)
			elapsed = 0
            for i = 1:num_chains
                m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
                data[:, i] = m[:, 2]
                # rm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv")
            end
            d = mean(data, dims=2)
            writedlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", d)
            performance_stats[:, al_step] = d
		elseif learning_algorithm == "VI"
			elapsed = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/elapsed_$(al_step).csv", ',')
            elapsed_stats[:, al_step] = elapsed
        end

        m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
        performance_data[1, al_step] = m[1, 2] #AcquisitionSize
        cd = readdlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
        performance_data[2, al_step] = cd[1, 2] #ClassDistEntropy
        performance_data[3, al_step] = m[2, 2] #Accuracy Score
        performance_data[4, al_step] = m[3, 2] #F1 
        performance_data[5, al_step] = acq_func
        performance_data[6, al_step] = temperature
        performance_data[7, al_step] = experiment
        performance_data[8, al_step] = al_step == 1 ? m[1, 2] : performance_data[8, al_step-1] + m[1, 2] #Cumulative Training Size
        performance_data[9, al_step] = prior_informativeness
        performance_data[10, al_step] = prior_variance
        performance_data[11, al_step] = likelihood_name

        for i in 1:n_output
            class_dist_data[i, al_step] = cd[i+1, 2]
            cum_class_dist_data[i, al_step] = al_step == 1 ? cd[i+1, 2] : cum_class_dist_data[i, al_step-1] + cd[i+1, 2]
        end
        cum_class_dist_ent[1, al_step] = normalized_entropy(cum_class_dist_data[:, al_step] ./ sum(cum_class_dist_data[:, al_step]), n_output)
    end
    if learning_algorithm == "MCMC"
        kpi = vcat(performance_data, class_dist_data, cum_class_dist_data, cum_class_dist_ent, performance_stats)
    elseif learning_algorithm == "VI"
        kpi = vcat(performance_data, class_dist_data, cum_class_dist_data, cum_class_dist_ent, elapsed_stats)
    end
    writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
    return kpi
end


function running_active_learning_ensemble(n_acq_steps, num_params, prior_std, pool, n_input, n_output, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, temperature, prior_informativeness, likelihood_name, learning_algorithm)
    # n_acq_steps = 10#round(Int, total_pool_samples / acquisition_size, RoundUp)
    prior = (zeros(num_params), prior_std)
    param_matrix, new_training_data = 0, 0
    new_noise_x = 0
    last_acc = 0
    best_acc = 0.5
    last_improvement = 0
    last_elapsed = 0
    benchmark_elapsed = 100.0
    new_pool = 0
    location_posterior = 0
    mcmc_init_params = 0
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
            new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(prior, pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, "Initial", mcmc_init_params, temperature, prior_informativeness, likelihood_name, learning_algorithm)
            mcmc_init_params = deepcopy(location_posterior)
            n_acq_steps = deepcopy(AL_iteration)
        elseif lastindex(new_pool[2]) >= acquisition_size
            if prior_informativeness == "UnInformedPrior"
                new_prior = prior
            else
                new_prior = (location_posterior, prior_std)
            end
            new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name, learning_algorithm)
            mcmc_init_params = deepcopy(location_posterior)
            n_acq_steps = deepcopy(AL_iteration)
        # elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
        #     if prior_informativeness == "UnInformedPrior"
        #         new_prior = prior
        #     else
        #         new_prior = (location_posterior, prior_std)
        #     end
        #     new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name, learning_algorithm)
        #     mcmc_init_params = deepcopy(location_posterior)
        #     println("Trained on last few samples remaining in the Pool")
        #     n_acq_steps = deepcopy(AL_iteration)
        end
    end
    return n_acq_steps
end

