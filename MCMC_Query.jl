"""
Returns new_pool, new_prior, independent_param_matrix, training_data
"""
function bnn_query(prior::Tuple, pool::Tuple, previous_training_data, input_size::Int, n_output::Int, param_matrix, map_matrix, al_step::Int, test_data, experiment::String, pipeline_name::String, acq_size_::Int, nsteps::Int, n_chains::Int, al_sampling::String, mcmc_init_params, temperature, class_balancing, prior_informativeness, prior_std_name, likelihood_name)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Array{Float32,2},Array{Float32,2},Array{Float32,2},Float32,Float32,Vector}
    println("$(al_sampling) with query no. ", al_step)
    # sigma, num_params = prior
    pool_x, pool_y = pool
    pool = vcat(pool_x, pool_y)
    pool_size = lastindex(pool_y)
    sampled_indices = 0
    if al_sampling == "Initial"
        sampled_indices = 1:acq_size_
    elseif al_sampling == "Random"
        sampled_indices = random_acquisition(pool_size, acq_size_)
    elseif al_sampling == "PowerBALD"
        pool_prediction_matrix = pool_predictions(pool_x, param_matrix, n_output)
        pool_scores = mapslices(x -> bald(x, n_output), pool_prediction_matrix, dims=[1, 3])
        bald_scores = map(x -> x[2], pool_scores[1, 1, :])
        sampled_indices = power_acquisition(bald_scores, acq_size_)
        # softmax_entropy = softmax_acquisition(entropy_scores, acq_size_)
        # var_ratio_scores = 1 .- pŷ_test
    elseif al_sampling == "BayesianUncertainty"
        pool_prediction_matrix = pool_predictions(pool_x, param_matrix, n_output)
        pool_scores = mapslices(x -> uncertainties(x, n_output), pool_prediction_matrix, dims=[1, 3])
        aleatoric_uncertainties = map(x -> x[2], pool_scores[1, 1, :])
        epistemic_uncertainties = map(x -> x[3], pool_scores[1, 1, :])
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/aleatoric_uncertainties_$(al_step).csv", summary_stats(aleatoric_uncertainties), ',')
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/epistemic_uncertainties_$(al_step).csv", summary_stats(epistemic_uncertainties), ',')
        most_unambiguous_samples = top_k_acquisition(aleatoric_uncertainties, round(Int, acq_size_ * 0.2))
        most_uncertain_samples = top_k_acquisition(epistemic_uncertainties, round(Int, acq_size_ * 0.8); descending=true, remove_zeros=true)
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/most_uncertain_samples_$(al_step).csv", most_uncertain_samples, ',')
        sampled_indices = union(most_unambiguous_samples, most_uncertain_samples)
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/samplsampled_indices_$(al_step).csv",sampled_indices, ',')
        #saving the uncertainties associated with the queried samples and their labels
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/sampled_indices_$(al_step)_stats.csv", [aleatoric_uncertainties[sampled_indices] epistemic_uncertainties[sampled_indices] pool_y[sampled_indices]], ',')
    elseif al_sampling == "PowerEntropy"
        pool_prediction_matrix = pool_predictions(pool_x, param_matrix, n_output)
        pool_scores = mapslices(x -> bald(x, n_output), pool_prediction_matrix, dims=[1, 3])
        entropy_scores = map(x -> x[1], pool_scores[1, 1, :])
        sampled_indices = power_acquisition(entropy_scores, acq_size_)
    elseif al_sampling == "Diversity"
        error("Diversity Sampling NOT IMPLEMENTED YET")
    elseif al_sampling == "PowerBayesian"
        bayesian_scores = pred_analyzer_multiclass(pool_x, param_matrix)
        sampled_indices = power_acquisition(bayesian_scores[2, :], acq_size_)
    elseif al_sampling == "QBC"
        bayesian_scores = pred_analyzer_multiclass(pool_x, param_matrix)
        sampled_indices = top_k_acquisition(bayesian_scores[2, :], acq_size_; descending=false)
    end


    new_acq_size_ = lastindex(sampled_indices)
    new_training_data = pool[:, sampled_indices]
    new_pool = pool[:, Not(sampled_indices)]
    if n_output == 2 && class_balancing == "BalancedBinaryAcquisition"
        new_training_data, leftovers = undersampling(new_training_data, positive_class_label=1, negative_class_label=2)
        if leftovers !== nothing
            new_pool = hcat(new_pool, leftovers)
            new_acq_size_ = new_acq_size_ - size(leftovers)[2]
        end
    end

    if new_acq_size_ == 0
        @info "Last Query Return Zero Samples, therefore Random Sampling!"
        sampled_indices = random_acquisition(pool_size, acq_size_)
        acq_size_ = lastindex(sampled_indices)
        new_training_data = pool[:, sampled_indices]
        new_pool = pool[:, Not(sampled_indices)]
        if n_output == 2
            new_training_data, leftovers = undersampling(new_training_data, positive_class_label=1, negative_class_label=2)
            if leftovers !== nothing
                new_pool = hcat(new_pool, leftovers)
                acq_size_ = acq_size_ - size(leftovers)[2]
            end
        end
    else
        acq_size_ = new_acq_size_
    end


    new_training_data_y = new_training_data[end, :]
    balance_of_acquired_batch = countmap(Int.(new_training_data_y))

    class_dist = Array{Int}(undef, n_output, 2)
    for i in 1:n_output
        try
            class_dist[i, 1] = i
            class_dist[i, 2] = balance_of_acquired_batch[i]
        catch
            class_dist[i, 1] = i
            class_dist[i, 2] = 0
        end
    end

    class_dist_ent = normalized_entropy(softmax(class_dist[:, 2]), n_output)

    if al_step == 1
        training_data = copy(new_training_data)
    else
		if experiment == "NonCumulative"
        	training_data = copy(new_training_data)
		else
			training_data = hcat(previous_training_data, new_training_data)
		end
    end

    training_data_x, training_data_y = training_data[1:input_size, :], training_data[end, :]

	#calculate the weights of the samples
	balance_of_training_data = countmap(Int.(training_data_y))
	sample_weights =  similar(training_data_y, Float32)
	nos_training = lastindex(training_data_y)
	for i = 1:nos_training
		sample_weights[i] = nos_training/balance_of_training_data[training_data_y[i]]
	end

    # println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
    training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
    # println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))

    if acq_size_ !== 0
        #Training on Acquired Samples and logging classification_performance
        independent_param_matrix, elapsed, independent_map_params = bayesian_inference(prior, training_data_xy, nsteps, n_chains, al_step, experiment, pipeline_name, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness)
    elseif acq_size_ == 0
        for i in 1:n_chains
            writedlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [0, 0, 0, 0, 0]], ',')
        end
        independent_param_matrix, elapsed, independent_map_params = param_matrix, 0, map_matrix
    end
    test_x, test_y = test_data
    predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
    predictions_map = pred_analyzer_multiclass(test_x, independent_map_params)
    writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv", predictions, ',')
    # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$(al_step)_map.csv", predictions_map, ',')
    ŷ_test = permutedims(Int.(predictions[1, :]))
    ŷ_test_map = permutedims(Int.(predictions_map[1, :]))
    # println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
    # pŷ_test = predictions[:,2]
    if n_output == 2
        acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
        acc_map, mcc_map, f1_map, fpr_map, prec_map, recall_map, threat_map, cm_map = performance_stats(test_y, ŷ_test_map)
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acq_size_, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step)_map.csv", [["Acquisition Size", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acq_size_, acc_map, mcc_map, f1_map, fpr_map, prec_map, recall_map, threat_map, cm_map]], ',')
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
        # println([["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acq_size_, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]])
    else
        acc = accuracy_multiclass(test_y, ŷ_test)
        acc_map = accuracy_multiclass(test_y, ŷ_test_map)
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Accuracy"] [acq_size_, acc]], ',')
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step)_map.csv", [["Acquisition Size", "Accuracy"] [acq_size_, acc_map]], ',')
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
        # println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acq_size_, balance_of_acquired_batch, acc]])
    end

    #Turning the posterior obtained after training on new samples into the new prior for the next iteration
    param_matrix_mean = vec(mean(independent_param_matrix, dims=1))
    # param_matrix_std = vec(std(independent_param_matrix, dims=1))
    # println("Euclidean distance between Prior and Posterior distribution's means is  ", euclidean(param_matrix_mean, prior[1]))
    # println("Euclidean distance between Prior and Posterior distribution's stds is ", euclidean(param_matrix_std, prior[2]))
    # println("Means of Prior and Posterior distribution's Means are  ", mean(prior[1]), " & ", mean(param_matrix_mean))	
    # println("Means of Prior and Posterior distribution's stds are  ", mean(prior[2]), " & ", mean(param_matrix_std))

    # writedlm("./Experiments/$(experiment)/$(pipeline_name)/log_distribution_changes/$al_step.csv", [["Euclidean distance between Prior and Posterior distribution's means is  ","Euclidean distance between Prior and Posterior distribution's stds is ", "Prior Mean", "Posterior distribution's Mean", "Priot StD", "Posterior distribution's std"] [euclidean(param_matrix_mean, prior[1]), euclidean(param_matrix_std, prior[2]), mean(prior[1]), mean(param_matrix_mean), mean(prior[2]), mean(param_matrix_std)]], ',')

    # new_prior = (param_matrix_mean, param_matrix_std)

    # println("size of training data is: ",size(training_data))
    # println("The dimenstions of the new_pool and param_matrix during AL step no. $al_step are:", size(new_pool), " & ", size(param_matrix))
    new_pool_tuple = (new_pool[1:input_size, :], permutedims(new_pool[end, :]))
    return new_pool_tuple, independent_param_matrix, independent_map_params, training_data, acc, elapsed, param_matrix_mean
end