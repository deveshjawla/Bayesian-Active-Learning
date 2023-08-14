"""
Returns means, stds, classifications, majority_voting, majority_conf

where

means, stds are the average logits and std for all the chains

classifications are obtained using threshold

majority_voting, majority_conf are averaged classifications of each chains, if the majority i.e more than 0.5 chains voted in favor then we have a vote in favor of 1. the conf here means how many chains on average voted in favor of 1. majority vote is just rounding of majority conf.
"""
function pred_analyzer(test_xs, test_ys, params_set, threshold)::Tuple{Array{Float32},Array{Float32},Array{Int},Array{Int},Array{Float32}}
    means = []
    stds = []
    classifications = []
    majority_voting = []
    majority_conf = []
    for (test_x, test_y) in zip(eachrow(test_xs), test_ys)
        predictions = []
        for theta in params_set
            model = feedforward(theta)
            # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
            ŷ = model(collect(test_x))
            append!(predictions, ŷ)
        end
        individual_classifications = map(x -> ifelse(x > threshold, 1, 0), predictions)
        majority_vote = ifelse(mean(individual_classifications) > 0.5, 1, 0)
        majority_conf_ = mean(individual_classifications)
        ensemble_pred_prob = mean(predictions) #average logit
        std_pred_prob = std(predictions)
        ensemble_class = ensemble_pred_prob > threshold ? 1 : 0
        append!(means, ensemble_pred_prob)
        append!(stds, std_pred_prob)
        append!(classifications, ensemble_class)
        append!(majority_voting, majority_vote)
        append!(majority_conf, majority_conf_)
    end

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return means, stds, classifications, majority_voting, majority_conf
end

using StatsBase

"""
Returns a tuple of {Prediction, Prediction probability}

Uses a simple argmax and percentage of samples in the ensemble respectively
"""
function pred_analyzer_multiclass(test_xs::Array{Float32, 2}, params_set::Array{Float32, 2})::Array{Float32, 2}
    n_samples = size(test_xs)[2]
	ensemble_size = size(params_set)[1]
	pred_matrix = Array{Float32}(undef, 2, n_samples)
    for i = 1:n_samples
		predictions = []
		for j = 1:ensemble_size
			model = feedforward(params_set[j,:])
				# make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
			ŷ = model(test_xs[:,i])
			# ŷ = feedforward(test_xs[:,i], params_set[j,:])#for the one with destructure
			# ŷ = feedforward(test_xs[:,i], params_set[j,:], network_shape)#for the one with unpack
			predicted_label = argmax(ŷ)
			append!(predictions, predicted_label)
		end
		count_map = countmap(predictions)
		# println(count_map)
		uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
		index_max = argmax(nUniques)
		prediction = uniques[index_max]
		pred_probability = maximum(nUniques) / sum(nUniques)
		# println(prediction, "\t", pred_probability)
		pred_matrix[:, i] = [prediction, pred_probability]
    end
    return pred_matrix
end


function convergence_stats(i::Int, chains, elapsed::Float32)::Tuple{Float32, Float32, Float32, Float32, Float32}
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
function pool_predictions(test_xs::Array{Float32, 2}, params_set::Array{Float32, 2}, n_output::Int)::Array{Float32, 3}
	n_samples = size(test_xs)[2]
	ensemble_size = size(params_set)[1]
	pred_matrix = Array{Float32}(undef, n_output, ensemble_size, n_samples)

    for i = 1:n_samples, j = 1:ensemble_size
            model = feedforward(params_set[j,:])
            # # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
            ŷ = model(test_xs[:,i])
            pred_matrix[:, j, i] = ŷ
			# if i==100 && j==100
			# 	println(ŷ)
			# end
    end
	# println(pred_matrix[:,100,100])
	return pred_matrix
end



function bayesian_inference(prior::Tuple, training_data::Tuple{Array{Float32, 2}, Array{Int, 2}}, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String)::Tuple{Array{Float32, 2}, Float32, Array{Float32, 2}}
	location, scale = prior
	@everywhere location = $location
	@everywhere scale = $scale
	# @everywhere network_shape = $network_shape
	nparameters = lastindex(location)
	@everywhere nparameters = $nparameters
	train_x, train_y = training_data
	@everywhere train_x = $train_x
	@everywhere train_y = $train_y
	println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
	# println(eltype(train_x), eltype(train_y))
	# println(mean(location), mean(scale))
	@everywhere model = bayesnnMVG(train_x, train_y, location, scale)
	chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains, progress = false)
	chains = chain_timed.value
	elapsed = Float32(chain_timed.time)
	println("It took $(elapsed) seconds to complete the $(nsteps) iterations")
	# writedlm("./$(experiment_name)/$(pipeline_name)/elapsed.txt", elapsed)
	θ = MCMCChains.group(chains, :θ).value

	gelman = gelmandiag(chains)
	psrf_psrfci =convert(Array, gelman)
	max_psrf = maximum(psrf_psrfci[:,1])
	writedlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_max_psrf.csv", max_psrf)

	# hyperprior = MCMCChains.group(chains, :input_hyperprior).value
	# θ_input = MCMCChains.group(chains, :θ_input).value
	# θ_hidden = MCMCChains.group(chains, :θ_hidden).value

	burn_in = Int(0.6*nsteps)
	n_indep_samples = Int((nsteps-burn_in) / 10)
	# hyperpriors_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, nparameters - lastindex(location))
	param_matrices_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, nparameters)
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

    	writedlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')

		# lPlot = Plots.plot(chains[:lp], title="Log Posterior", label=:none)
		df = DataFrame(chains)
		df[!, :chain] = categorical(df.chain)
		lPlot = Gadfly.plot(df, y=:lp, x=:iteration, Geom.line, color=:chain, Guide.title("Log Posterior"), Coord.cartesian(xmin=df.iteration[1], xmax=df.iteration[1] + nsteps))
		# plt= Plots.plot(lPlot, size=(1600, 600))
		# Plots.savefig(plt, "./$(experiment_name)/$(pipeline_name)/convergence_statistics/chain_$(i).png")
		lPlot |> PNG("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_lp.png", 800pt, 600pt)

		lp, maxInd = findmax(chains[:lp])
		params, internals = chains.name_map
		bestParams = map(x -> chains[x].data[maxInd], params[1:nparameters])
		map_params_accumulated[i, :] = bestParams
		# println(oob_rhat)

		# hyperpriors_accumulated[(i-1)*size(independent_hyperprior_matrix)[1]+1:i*size(independent_hyperprior_matrix)[1],:] = independent_hyperprior_matrix
		param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1],:] = independent_param_matrix
    end
	# hyperpriors_mean = mean(hyperpriors_accumulated, dims = 1)
    # writedlm("./$(experiment_name)/$(pipeline_name)/hyperpriors/$al_step.csv", hyperpriors_mean, ',')
	writedlm("./$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$(al_step)_MAP.csv", mean(map_params_accumulated, dims=1), ',')
    writedlm("./$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
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
# 	# writedlm("./$(experiment_name)/$(pipeline_name)/elapsed.txt", elapsed)
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

#     	writedlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')
# 		# println(oob_rhat)

# 		param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1],:] = independent_param_matrix
#     end
#     writedlm("./$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
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
