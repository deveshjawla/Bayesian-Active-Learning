
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


function convergence_stats(i::Int, chains, elapsed::Float32)::Tuple{Float32,Float32,Float32,Float32,Float32}
    ch = chains[:, :, i]
    summaries, quantiles = describe(ch)
    large_rhat = count(>=(1.01), sort(summaries[:, :rhat]))
    small_rhat = count(<=(0.99), sort(summaries[:, :rhat]))
    oob_rhat = large_rhat + small_rhat
    avg_acceptance_rate = mean(ch[:acceptance_rate])
    total_numerical_error = sum(ch[:numerical_error])
    avg_ess = mean(summaries[:, :ess_per_sec])
    # println(describe(summaries[:, :mean]))
    return elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess
end


"""
Returns a matrix of dims (n_output, ensemble_size, n_samples)
"""
function pool_predictions(test_xs::Array{Float32,2}, params_set::Array{Float32,2}; reconstruct=nothing)::Array{Float32,3}
    if isnothing(reconstruct)
        nets = map(feedforward, eachrow(params_set))
    else
        nets = map(reconstruct, eachrow(params_set))
    end
    predictions_nets = map(net -> net(test_xs), nets)
    pred_matrix = cat(predictions_nets..., dims=3)
    return pred_matrix
end



function bayesian_inference(prior::Tuple, training_data::Tuple{Array{Float32,2},Array{Int,2}}, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness)::Tuple{Array{Float32,2},Float32,Array{Float32,2}}
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
    elseif likelihood_name == "WeightedLikelihood" && temperature === nothing
        @everywhere model = classweightedBNN(train_x, train_y, location, scale, sample_weights)
    else
        nothing
        # @everywhere model = BNN(train_x, train_y, location, scale)
    end

    if al_step == 1 || prior_informativeness == "NoInit"
        chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains, progress=false)
    else
        chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains, init_params=repeat([mcmc_init_params], n_chains), progress=false)
    end
    chains = chain_timed.value
    elapsed = Float32(chain_timed.time)
    println("It took $(elapsed) seconds to complete the $(nsteps) iterations")
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/elapsed.txt", elapsed)
    θ = MCMCChains.group(chains, :θ).value

    gelman = gelmandiag(chains)
    psrf_psrfci = convert(Array, gelman)
    max_psrf = maximum(psrf_psrfci[:, 1])
    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_max_psrf.csv", max_psrf)

    # hyperprior = MCMCChains.group(chains, :input_hyperprior).value
    # θ_input = MCMCChains.group(chains, :θ_input).value
    # θ_hidden = MCMCChains.group(chains, :θ_hidden).value

    burn_in = 0#Int(0.6 * nsteps)
    n_indep_samples = Int((nsteps - burn_in) / 10)
    # hyperpriors_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, nparameters - lastindex(location))
    param_matrices_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, nparameters)
    map_params_accumulated = Array{Float32}(undef, n_chains, nparameters)

    for i in 1:n_chains
        params_set = collect.(eachrow(θ[:, :, i]))
        # hyperprior_set = collect.(eachrow(hyperprior[:, :, i]))
        # params_set_input = collect.(eachrow(θ_input[:, :, i]))
        # params_set_hidden = collect.(eachrow(θ_hidden[:, :, i]))
        param_matrix = mapreduce(permutedims, vcat, params_set)
        # hyperprior_matrix = mapreduce(permutedims, vcat, hyperprior_set)
        # param_matrix_input = mapreduce(permutedims, vcat, params_set_input)
        # param_matrix_hidden = mapreduce(permutedims, vcat, params_set_hidden)
        # param_matrix = hcat(param_matrix_input, param_matrix_hidden)

        # independent_hyperprior_matrix = Array{Float32}(undef, n_indep_samples, nparameters - lastindex(location))
        independent_param_matrix = Array{Float32}(undef, n_indep_samples, nparameters)
        for i in 1:nsteps-burn_in
            if i % 10 == 0
                # independent_hyperprior_matrix[Int((i) / 10), :] = hyperprior_matrix[i+burn_in, :]
                independent_param_matrix[Int((i) / 10), :] = param_matrix[i+burn_in, :]
            end
        end

        elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chains, elapsed)

        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')

        # # lPlot = Plots.plot(chains[:lp], title="Log Posterior", label=:none)
        # df = DataFrame(chains)
        # df[!, :chain] = categorical(df.chain)
        # lPlot = Gadfly.plot(df, y=:lp, x=:iteration, Geom.line, color=:chain, Guide.title("Log Posterior"), Coord.cartesian(xmin=df.iteration[1], xmax=df.iteration[1] + nsteps)) # need to change :acceptance_rate to Log Posterior
        # # plt= Plots.plot(lPlot, size=(1600, 600))
        # # Plots.savefig(plt, "./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/chain_$(i).pdf")
        # lPlot |> PDF("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_lp.pdf", 800pt, 600pt)

        # lp, maxInd = findmax(chains[:lp])
        # params, internals = chains.name_map
        # bestParams = map(x -> chains[x].data[maxInd], params[1:nparameters])
        # map_params_accumulated[i, :] = bestParams
        # println(oob_rhat)

        # hyperpriors_accumulated[(i-1)*size(independent_hyperprior_matrix)[1]+1:i*size(independent_hyperprior_matrix)[1],:] = independent_hyperprior_matrix
        param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1], :] = independent_param_matrix
    end
    # hyperpriors_mean = mean(hyperpriors_accumulated, dims = 1)
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/hyperpriors/$al_step.csv", hyperpriors_mean, ',')
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$(al_step)_MAP.csv", mean(map_params_accumulated, dims=1), ',')
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    return param_matrices_accumulated, elapsed, map_params_accumulated
end

# function bayesian_inference_single_core(prior, training_data, nsteps, n_chains, al_step, pipeline_name)
# 	location_prior, scale_prior = prior
# 	train_x, train_y = training_data
# 	# println("Checking dimensions of train_x and train_y just before training:", train_x[1,1], " & ", train_y[1,1])
# 	model = bayesnnMVG(train_x, train_y, total_num_params)
# 	chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains)
# 	chains = chain_timed.value
# 	elapsed = chain_timed.time
# 	# writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/elapsed.txt", elapsed)
# 	θ = MCMCChains.group(chains, :θ).value

# 	burn_in = Int(0.6*nsteps)
# 	n_indep_samples = Int((nsteps-burn_in) / 10)
# 	param_matrices_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, total_num_params)
#     for i in 1:n_chains
# 		params_set = collect.(eachrow(θ[:, :, i]))
#     	param_matrix = mapreduce(permutedims, vcat, params_set)

# 		independent_param_matrix = Array{Float32}(undef, n_indep_samples, total_num_params)
# 		for i in 1:nsteps-burn_in
# 			if i % 10 == 0
# 				independent_param_matrix[Int((i) / 10), :] = param_matrix[i+burn_in, :]
# 			end
# 		end

# 		elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chains, elapsed)

#     	writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')
# 		# println(oob_rhat)

# 		param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1],:] = independent_param_matrix
#     end
#     writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
# 	return param_matrices_accumulated
# end


# Tools to Examine Chains
# summaries, quantiles = describe(chains);
# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chains[:lp])
# i = i.I[1]
# θ[i, :]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chains, :θ).value


using StatsBase: countmap
function majority_voting(predictions::AbstractVector)::Vector{Float64}
    count_map = countmap(predictions)
    # println(count_map)
    uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
    index_max = argmax(nUniques)
    prediction = uniques[index_max]
    pred_probability = maximum(nUniques) / sum(nUniques)
    return [prediction, 1 - pred_probability] # 1 - pred_probability => give the unceratinty
end

"""
EDLC Uncertainty
"""
edlc_uncertainty(α) = first(size(α)) ./ sum(α, dims=1)

function normalized_entropy(prob_vec::Vector, n_output)::Float64
    sum_probs = sum(prob_vec)
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum_probs < 0.99999 || sum_probs > 1.00001
        return error("sum(prob_vec) is not 1 BUT $(sum_probs) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log.(prob_vec))) / log(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::Matrix, n_output)::Float32
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    return H
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

The H is the Total Uncertainty
E_H is the Aleatoric
H - E_H is the Epistemic

Returns : H (macro_entropy), E_H (mean of Entropies) is the Aleatoric, H - E_H(Epistemic unceratinty)
"""
function bald(prob_matrix::Matrix, n_output)
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H - E_H
end

function pred_analyzer_multiclass(test_xs::Array{Float64,2}, param_matrix::Array{Float64,2}; noise_set=nothing, reconstruct=nothing)::Array{Float64,2}
    if isnothing(reconstruct)
        nets = map(feedforward, eachrow(param_matrix))
    else
        nets = map(reconstruct, eachrow(param_matrix))
    end
	if isnothing(noise_set)
		# predictions_nets = map(x -> softmax(x(test_xs)), nets)
		predictions_nets = map(net -> net(test_xs) ./sum(x(test_xs), dims=1), nets)
	else
    	predictions_nets = map((x, y) -> x(test_xs .+ y) ./ sum(x(test_xs .+ y), dims=1), nets, noise_set)
	end

    # Determine the size of the final 3D array
    num_matrices = lastindex(predictions_nets)
    rows, cols = size(predictions_nets[1])

    # Preallocate the 3D array
    pred_matrix = Array{Float64}(undef, rows, cols, num_matrices)

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
function pred_analyzer_regression(test_xs::Array{Float32,2}, params_set::Array{Float32,2})::Tuple{Array{Float32,2},Array{Float32,2}}
    if isnothing(reconstruct)
        nets = map(feedforward, eachrow(param_matrix))
    else
        nets = map(reconstruct, eachrow(params_set))
    end
    predictions_nets = map(net -> net(test_xs), nets)
    predictions = permutedims(reduce(vcat, predictions_nets))
    pred_matrix_mean = mapslices(mean, predictions, dims=2)
    pred_matrix_std = mapslices(std, predictions, dims=2)
    return pred_matrix_mean, pred_matrix_std
end


# function variances_ratio(predictions::Vector)
#     count_map = countmap(predictions)
#     # println(count_map)
#     uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
#     index_max = argmax(nUniques)
#     prediction = uniques[index_max]
#     pred_probability = maximum(nUniques) / sum(nUniques)
#     return 1 - pred_probability
# end

