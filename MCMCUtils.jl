
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
    activations_weights_plot |> PDF("$(directory_plots)/activations_weights.pdf", dpi=600)
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

function mcmc_inference(prior::Tuple, training_data::Tuple{Matrix{Float32}, Matrix{Int64}}, n_input, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness)::Tuple{Array{Float32,2},Vector{Vector{Float32}}, Float32}
    location, scale = prior
    @everywhere location = $location
    @everywhere scale = $scale
    @everywhere mcmc_init_params = $mcmc_init_params
    # @everywhere network_shape = $network_shape
    nparameters = lastindex(location)
    @everywhere nparameters = $nparameters
    train_x, train_y = training_data
    @everywhere train_x = $train_x
    @everywhere train_y = $train_y
    @everywhere sample_weights = $sample_weights
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # println(eltype(train_x), eltype(train_y))
    # println(mean(location), mean(scale))
    if temperature isa Number && likelihood_name == "TemperedLikelihood"
        @everywhere model = temperedBNN(train_x, train_y, location, scale, temperature)
    elseif likelihood_name == "WeightedLikelihood" && temperature == "CWL"
        @everywhere model = classweightedBNN(train_x, train_y, location, scale, sample_weights)
    else
        @everywhere model = softmax_bnn_noise_x(train_x, train_y, location, scale)
    end

    if al_step == 1 || prior_informativeness == "NoInit"
        chain_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff()), MCMCDistributed(), nsteps, n_chains, progress=false)
    else
        chain_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff()), MCMCDistributed(), nsteps, n_chains, init_params=repeat([mcmc_init_params], n_chains), progress=false)
    end
    chains = chain_timed.value
    elapsed = Float32(chain_timed.time)
    println("It took $(elapsed) seconds to complete the $(nsteps) iterations")
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/elapsed.csv", elapsed)
    θ = MCMCChains.group(chains, :θ).value
    noise_x = MCMCChains.group(chains, :noise_x).value

	acr_lags = [5, 10, 50, 100]
    acr = MCMCChains.autocor(chains, lags=acr_lags)
	autocorrelations_df = DataFrame(acr)
	means_acr = map(mean, eachcol(autocorrelations_df[:, 2:end]))

	min_acr = Inf
	min_acr_idx=1
	for (i,j) in enumerate(means_acr)
		if j < min_acr && j > 0
			println(i,j, min_acr)
			min_acr = j
			min_acr_idx = i
		end
	end

	mc_autocor_lag = acr_lags[min_acr_idx]
	@info "Autocorrelation IDX" means_acr

    burn_in = 0#Int(0.6 * nsteps)
    n_indep_samples = round(Int, (nsteps - burn_in) / mc_autocor_lag)
    # hyperpriors_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, nparameters - lastindex(location))
    param_matrices_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, nparameters)
    noise_x_vectors_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, n_input)

    for i in 1:n_chains
        params_set = collect.(eachrow(θ[:, :, i]))
        noise_set_x = collect.(eachrow(noise_x[:, :, i]))
        param_matrix = mapreduce(permutedims, vcat, params_set)
        noise_x_matrix = mapreduce(permutedims, vcat, noise_set_x)

        independent_param_matrix = Array{Float32}(undef, n_indep_samples, nparameters)
        independent_noise_x_matrix = Array{Float32}(undef, n_indep_samples, n_input)
        for j in 1:nsteps-burn_in
            if j % mc_autocor_lag == 0
                independent_param_matrix[Int((j) / mc_autocor_lag), :] = param_matrix[j+burn_in, :]
                independent_noise_x_matrix[Int((j) / mc_autocor_lag), :] = noise_x_matrix[j+burn_in, :]
            end
        end

        elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf = convergence_stats(i, chains, elapsed)

        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "max_psrf"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf]], ',')

        # # lPlot = plot(chains[:lp], title="Log Posterior", label=:none)
        # df = DataFrame(chains)
        # df[!, :chain] = categorical(df.chain)
        # lPlot = Gadfly.plot(df, y=:lp, x=:iteration, Geom.line, color=:chain, Guide.title("Log Posterior"), Coord.cartesian(xmin=df.iteration[1], xmax=df.iteration[1] + nsteps)) # need to change :acceptance_rate to Log Posterior
        # # plt= plot(lPlot, size=(1600, 600))
        # # savefig(plt, "./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/chain_$(i).pdf")
        # lPlot |> PDF("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_lp.pdf", 800pt, 600pt)

        # lp, maxInd = findmax(chains[:lp])
        # params, internals = chains.name_map
        # bestParams = map(x -> chains[x].data[maxInd], params[1:nparameters])
        # map_params_accumulated[i, :] = bestParams
        # println(oob_rhat)
        param_matrices_accumulated[(i-1)*size(independent_param_matrix, 1)+1:i*size(independent_param_matrix, 1), :] = independent_param_matrix
        noise_x_vectors_accumulated[(i-1)*size(independent_noise_x_matrix, 1)+1:i*size(independent_noise_x_matrix, 1), :] = independent_noise_x_matrix
    end
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$(al_step)_MAP.csv", mean(map_params_accumulated, dims=1), ',')
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
	noise_x_vectors_accumulated = collect.(Float32, eachrow(noise_x_vectors_accumulated))
    return param_matrices_accumulated, noise_x_vectors_accumulated, elapsed
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


function collecting_stats_active_learning_experiments_regression(n_acq_steps, experiment, pipeline_name, num_chains, n_output)
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
    return kpi
end

function collecting_stats_active_learning_experiments_classification(n_acq_steps, experiment, pipeline_name, num_chains, n_output)
    performance_stats = Array{Float32}(undef, 6, n_acq_steps)
    class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    cum_class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    performance_data = Array{Any}(undef, 11, n_acq_steps) #dims=(features, samples(i))
    cum_class_dist_ent = Array{Float32}(undef, 1, n_acq_steps)

    for al_step = 1:n_acq_steps
        data = Array{Any}(undef, 6, num_chains)
        for i = 1:num_chains
            m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
            data[:, i] = m[:, 2]
            # rm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv")
        end
        d = mean(data, dims=2)
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain.csv", d)
        performance_stats[:, al_step] = d

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
    kpi = vcat(performance_data, class_dist_data, cum_class_dist_data, cum_class_dist_ent, performance_stats)
    writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
    return kpi
end


function running_active_learning_ensemble(n_acq_steps, num_params, prior_std, pool, n_input, n_output, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, temperature, prior_informativeness, likelihood_name)
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
            new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(prior, pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, "Initial", mcmc_init_params, temperature, prior_informativeness, likelihood_name)
            mcmc_init_params = deepcopy(location_posterior)
            n_acq_steps = deepcopy(AL_iteration)
        elseif lastindex(new_pool[2]) > acquisition_size
            if prior_informativeness == "UnInformedPrior"
                new_prior = prior
            else
                new_prior = (location_posterior, prior_std)
            end
            new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, acquisition_size, num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name)
            mcmc_init_params = deepcopy(location_posterior)
            n_acq_steps = deepcopy(AL_iteration)
        elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
            if prior_informativeness == "UnInformedPrior"
                new_prior = prior
            else
                new_prior = (location_posterior, prior_std)
            end
            new_pool, param_matrix, new_noise_x, new_training_data, last_acc, last_elapsed, location_posterior = bnn_query(new_prior, new_pool, new_training_data, n_input, n_output, param_matrix, new_noise_x, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), num_mcsteps, num_chains, acq_func, mcmc_init_params, temperature, prior_informativeness, likelihood_name)
            mcmc_init_params = deepcopy(location_posterior)
            println("Trained on last few samples remaining in the Pool")
            n_acq_steps = deepcopy(AL_iteration)
        end
    end
    return n_acq_steps
end


function mean_std_by_variable(group, group_by::Symbol, measurable::Symbol, variable::Symbol)
    mean_std = DataFrames.combine(groupby(group, variable), measurable => mean, measurable => std)
    group_name = first(group[!, group_by])
    mean_std[!, group_by] = repeat([group_name], nrow(mean_std))
    CSV.write("./Experiments/$(experiment)/mean_std_$(group_name)_$(measurable)_$(variable).csv", mean_std)
end
function mean_std_by_group(df_folds, group_by::Symbol, variable::Symbol; list_measurables=[:MSE, :Elapsed, :MAE, :AcceptanceRate, :NumericalErrors])
    for group in groupby(df_folds, group_by)
        for m in list_measurables
            mean_std_by_variable(group, group_by, m, variable)
        end
    end
end

function auc_per_fold(fold::Int, df::DataFrame, group_by::Symbol, measurement1::Symbol, measurement2::Symbol)
    aucs_acc = []
    aucs_t = []
    list_compared = []
    list_total_training_samples = []
    for i in groupby(df, group_by)
        acc_ = i[!, measurement1]
        time_ = i[!, measurement2]
        # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
        n_aocs_samples = lastindex(acc_)
        total_training_samples = last(i[!, :CumTrainedSize])
        println(total_training_samples)
        push!(list_total_training_samples, total_training_samples)
        auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
        auc_t = mean(time_[1:n_aocs_samples] .- 0.0) / total_training_samples
        push!(list_compared, first(i[!, group_by]))
        append!(aucs_acc, (auc_acc))
        append!(aucs_t, auc_t)
    end
    min_total_samples = minimum(list_total_training_samples)
    df = DataFrame(group_by => list_compared, measurement1 => min_total_samples .* (aucs_acc), measurement2 => min_total_samples .* aucs_t)
    CSV.write("./Experiments/$(experiment)/auc_$(fold).csv", df)
end
function auc_mean(n_folds, experiment, group_by::Symbol, measurement1::Symbol, measurement2::Symbol)
    df = DataFrame()
    for fold = 1:n_folds
        df_ = CSV.read("./Experiments/$(experiment)/auc_$(fold).csv", DataFrame, header=1)
        df = vcat(df, df_)
    end

    mean_auc = combine(groupby(df, group_by), measurement1 => mean, measurement1 => std, measurement2 => mean, measurement2 => std)

    CSV.write("./Experiments/$(experiment)/mean_auc.csv", mean_auc)
end

function plotting_measurable_variable(experiment, groupby::Symbol, list_group_names, dataset, variable::Symbol, measurable::Symbol, measurable_mean::Symbol, measurable_std::Symbol, normalised_measurable::Bool)
    width = 6inch
    height = 6inch
    set_default_plot_size(width, height)
    theme = Theme(major_label_font_size=16pt, minor_label_font_size=14pt, key_title_font_size=14pt, key_label_font_size=12pt, key_position=:none, colorkey_swatch_shape=:circle, key_swatch_size=12pt)
    Gadfly.push_theme(theme)

    df = DataFrame()
    for group_name in list_group_names
        df_ = CSV.read("./Experiments/$(experiment)/mean_std_$(group_name)_$(measurable)_$(variable).csv", DataFrame, header=1)
        # df_[!, :AcquisitonFunction] .= repeat([group_name], nrow(df_))
        df = vcat(df, df_)
    end
    if normalised_measurable
        y_ticks = collect(0:0.1:1.0)
        fig1a = Gadfly.plot(df, x=variable, y=measurable_mean, color=groupby, ymin=df[!, measurable_mean] - df[!, measurable_std], ymax=df[!, measurable_mean] + df[!, measurable_std], Geom.point, Geom.line, Geom.ribbon, Geom.hline(color=["red"], size=[0.5mm]), Guide.ylabel(String(measurable)), Guide.xlabel(String(variable)), yintercept=[0.5], Guide.yticks(ticks=y_ticks), Coord.cartesian(xmin=xmin = df[!, variable][1], ymin=0.0, ymax=1.0))
    else
        fig1a = Gadfly.plot(df, x=variable, y=measurable_mean, color=groupby, ymin=df[!, measurable_mean] - df[!, measurable_std], ymax=df[!, measurable_mean] + df[!, measurable_std], Geom.point, Geom.line, Geom.ribbon, Guide.ylabel(String(measurable)), Guide.xlabel(String(variable)))
    end
    fig1a |> PDF("./Experiments/$(experiment)/$(measurable)_$(variable)_$(dataset)_$(experiment)_folds.pdf", dpi=600)
end
