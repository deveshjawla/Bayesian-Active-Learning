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



function bayesian_inference(prior::Tuple, training_data::Tuple{Array{Float32, 2}, Array{Int, 2}}, nsteps::Int, n_epochs::Int, al_step::Int, experiment_name::String, pipeline_name::String)::Tuple{Array{Float32, 2}, Float32}
	sigma, nparameters = prior
	# @everywhere network_shape = $network_shape
	@everywhere nparameters = $nparameters
	train_x, train_y = training_data
	@everywhere train_x = $train_x
	@everywhere train_y = $train_y
	println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
	# println(eltype(train_x), eltype(train_y))
	@everywhere model = bayesnnMVG(train_x, train_y, sigma, nparameters)
	q = Variational.meanfield(model)

	μ = randn(length(q))
	ω = exp.(-1 .* ones(length(q)))

	q = AdvancedVI.update(q, μ, ω)

	# @info("Beginning training loop...")
	best_elbo = -Inf32
	last_improvement = 0
	q̂ = 0
	elapsed = 0
	param_matrices_accumulated = 0
	for epoch_idx = 1:n_epochs
		advi = ADVI(10, nsteps)
		chain_timed = @timed vi(model, advi, q)
		q_hat = chain_timed.value
		elapsed = Float32(chain_timed.time)
		elbo_score = elbo(advi, q, model, 1000)
		
		# If this is the minimum loss we've seen so far, save the model out
		if elbo_score > best_elbo
			@info(" -> New best ELBO! Saving model weights")
			q̂ = q_hat
			best_elbo = elbo_score
			last_improvement = epoch_idx
			param_matrices_accumulated = permutedims(rand(q_hat, 5000))
		end

		if epoch_idx - last_improvement >= 5
			@warn(" -> We're calling this converged.")
			break
		end
	end
	# writedlm("./$(experiment_name)/$(pipeline_name)/elapsed.txt", elapsed)
	
	writedlm("./$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step).csv", [["elapsed", "ELBO"] [elapsed, best_elbo]], ',')
		# println(oob_rhat)

    writedlm("./$(experiment_name)/$(pipeline_name)/posterior_dist/$al_step.csv", [q̂.dist.m q̂.dist.σ], ',')
	return param_matrices_accumulated, elapsed
end