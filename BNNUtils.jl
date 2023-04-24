"""
Returns means, stds, classifications, majority_voting, majority_conf

where

means, stds are the average logits and std for all the chains

classifications are obtained using threshold

majority_voting, majority_conf are averaged classifications of each chain, if the majority i.e more than 0.5 chains voted in favor then we have a vote in favor of 1. the conf here means how many chains on average voted in favor of 1. majority vote is just rounding of majority conf.
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
function pred_analyzer_multiclass(test_xs, params_set)::Array{Float32, 2}
    n_samples = size(test_xs)[2]
	ensemble_size = size(params_set)[1]
	pred_matrix = Array{Float32}(undef, 2, n_samples)
    for i = 1:n_samples
		predictions = []
		for j = 1:ensemble_size
			model = feedforward(params_set[j,:])
				# make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
			ŷ = model(test_xs[:,i])
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

using EvalMetrics
function performance_stats(ground_truth_, predictions_)
	ground_truth = deepcopy(Int.(vec(ground_truth_)))
    predictions = deepcopy(Int.(vec(predictions_)))
    ground_truth[ground_truth.==2] .= 0
    predictions[predictions.==2] .= 0
    f1 = f1_score(ground_truth, predictions)
    mcc = matthews_correlation_coefficient(ground_truth, predictions)
    acc = accuracy(ground_truth, predictions)
    fpr = false_positive_rate(ground_truth, predictions)
    # fnr = fnr(ground_truth, predictions)
    # tpr = tpr(ground_truth, predictions)
    # tnr = tnr(ground_truth, predictions)
    prec = precision(ground_truth, predictions)
    recall = true_positive_rate(ground_truth, predictions)
	threat_score = EvalMetrics.threat_score(ground_truth, predictions)
	cm = ConfusionMatrix(ground_truth, predictions)
    return acc, mcc, f1, fpr, prec, recall, threat_score, cm
end

function accuracy_multiclass(true_labels, predictions)
    return mean(true_labels .== predictions)
end

function convergence_stats(i, chain, elapsed)
    ch = chain[:, :, i]
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
function pool_predictions(test_xs, params_set, n_output)::Array{Float32, 3}
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


# function bayesian_inference(prior, training_data, nsteps, n_chains, al_step, name_exp)
# 	location_prior, scale_prior = prior
# 	@everywhere location_prior = $location_prior
# 	@everywhere scale_prior = $scale_prior
# 	train_x, train_y = training_data
# 	@everywhere train_x = $train_x
# 	@everywhere train_y = $train_y
# 	# println("Checking dimensions of train_x and train_y just before training:", train_x[1,1], " & ", train_y[1,1])
# 	@everywhere model = bayesnnMVG(train_x, train_y, location_prior, scale_prior)
# 	chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains)
# 	chain = chain_timed.value
# 	elapsed = chain_timed.time
# 	# writedlm("./experiments/$(name_exp)/elapsed.txt", elapsed)
# 	θ = MCMCChains.group(chain, :θ).value

# 	burn_in = Int(0.6*nsteps)
# 	n_indep_samples = Int((nsteps-burn_in) / 10)
# 	param_matrices_accumulated = Array{Float64}(undef, n_chains*n_indep_samples, total_num_params)
#     for i in 1:n_chains
# 		params_set = collect.(eachrow(θ[:, :, i]))
#     	param_matrix = mapreduce(permutedims, vcat, params_set)

# 		independent_param_matrix = Array{Float64}(undef, n_indep_samples, total_num_params)
# 		for i in 1:nsteps-burn_in
# 			if i % 10 == 0
# 				independent_param_matrix[Int((i) / 10), :] = param_matrix[i+burn_in, :]
# 			end
# 		end

# 		elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chain, elapsed)

#     	writedlm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')
# 		# println(oob_rhat)

# 		param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1],:] = independent_param_matrix
#     end
#     writedlm("./experiments/$(name_exp)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
# 	return param_matrices_accumulated
# end

function bayesian_inference(prior, training_data, nsteps, n_chains, al_step, name_exp)
	location_prior, scale_prior = prior
	train_x, train_y = training_data
	# println("Checking dimensions of train_x and train_y just before training:", train_x[1,1], " & ", train_y[1,1])
	model = bayesnnMVG(train_x, train_y, location_prior, scale_prior)
	chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains)
	chain = chain_timed.value
	elapsed = chain_timed.time
	# writedlm("./experiments/$(name_exp)/elapsed.txt", elapsed)
	θ = MCMCChains.group(chain, :θ).value

	burn_in = Int(0.6*nsteps)
	n_indep_samples = Int((nsteps-burn_in) / 10)
	param_matrices_accumulated = Array{Float64}(undef, n_chains*n_indep_samples, total_num_params)
    for i in 1:n_chains
		params_set = collect.(eachrow(θ[:, :, i]))
    	param_matrix = mapreduce(permutedims, vcat, params_set)

		independent_param_matrix = Array{Float64}(undef, n_indep_samples, total_num_params)
		for i in 1:nsteps-burn_in
			if i % 10 == 0
				independent_param_matrix[Int((i) / 10), :] = param_matrix[i+burn_in, :]
			end
		end

		elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chain, elapsed)

    	writedlm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')
		# println(oob_rhat)

		param_matrices_accumulated[(i-1)*size(independent_param_matrix)[1]+1:i*size(independent_param_matrix)[1],:] = independent_param_matrix
    end
    writedlm("./experiments/$(name_exp)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
	return param_matrices_accumulated
end
