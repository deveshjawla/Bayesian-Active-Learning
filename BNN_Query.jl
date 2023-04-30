"""
Returns new_pool, new_prior, independent_param_matrix
"""
function query_function(prior, pool, previous_training_data, input_size, n_output, param_matrix, al_step, test_data, name_exp, acquisition_size, nsteps, n_chains, al_sampling)
	println("$(al_sampling) with acquisition step no. ", al_step)

	pool_x, pool_y = pool
	pool = vcat(pool_x, pool_y)
	pool_size = lastindex(pool_y)
	sampled_indices = 0
	if al_sampling == "Random"
		sampled_indices = initial_random_acquisition(pool_size, acquisition_size)
	elseif al_sampling == "PowerBALD"
		pool_prediction_matrix = pool_predictions(pool_x, param_matrix, n_output)
		pool_scores = mapslices(x->bald(x,n_output), pool_prediction_matrix, dims=[1,2])
		bald_scores = map(x->x[2], pool_scores[1,1,:])
		sampled_indices = power_acquisition(bald_scores, acquisition_size)
		# softmax_entropy = softmax_acquisition(entropy_scores, acquisition_size)
		# var_ratio_scores = 1 .- pŷ_test
	elseif al_sampling == "Diversity"
		sampled_indices = initial_random_acquisition(pool_size, acquisition_size)
	end

	new_training_data = pool[:, sampled_indices]
	new_pool = pool[:, Not(sampled_indices)]

	new_training_data_y = new_training_data[end, :]
	balance_of_acquired_batch = countmap(Int.(new_training_data_y))

	class_dist = Array{Int}(undef, n_output, 2)
	for i in 1:n_output
		try 
			class_dist[i,1]= i
			class_dist[i,1] = balance_of_acquired_batch[i]
		catch
			class_dist[i,1]= i
			class_dist[i,1] = 0
		end
	end

	class_dist_ent = normalized_entropy(softmax(class_dist[:,2]), n_output)

	training_data = new_training_data
	# training_data = hcat(previous_training_data, new_training_data)

	training_data_x, training_data_y = training_data[1:input_size, :], training_data[end, :]
	# println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
	training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
	# println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))

	#Training on Acquired Samples and logging classification_performance
	independent_param_matrix = bayesian_inference(prior, training_data_xy, nsteps, n_chains, al_step, name_exp)
	
	test_x, test_y = test_data
	predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
    writedlm("./$(experiment_name)/$(pipeline_name)/predictions/$al_step.csv", predictions, ',')
	ŷ_test = permutedims(Int.(predictions[1,:]))
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./$(experiment_name)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
		writedlm("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
		# println([["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]])
	else
		acc = accuracy_multiclass(test_y, ŷ_test)
		writedlm("./$(experiment_name)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size","Accuracy"]], ',')
		writedlm("./$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
		# println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]])
	end
	#Turning the posterior obtained after training on new samples into the new prior for the next iteration
    param_matrix_mean = vec(mean(independent_param_matrix, dims=1))
    param_matrix_std = vec(std(independent_param_matrix, dims=1))
    # println("Euclidean distance between Prior and Posterior distribution's means is  ", euclidean(param_matrix_mean, prior[1]))
    # println("Euclidean distance between Prior and Posterior distribution's stds is ", euclidean(param_matrix_std, prior[2]))
	# println("Means of Prior and Posterior distribution's Means are  ", mean(prior[1]), " & ", mean(param_matrix_mean))	
	# println("Means of Prior and Posterior distribution's stds are  ", mean(prior[2]), " & ", mean(param_matrix_std))

	writedlm("./$(experiment_name)/$(pipeline_name)/log_distribution_changes/$al_step.csv", [["Euclidean distance between Prior and Posterior distribution's means is  ","Euclidean distance between Prior and Posterior distribution's stds is ", "Means of Prior", "and Posterior distribution's Means are  ", "Means of Prior", "and Posterior distribution's stds are  "] [euclidean(param_matrix_mean, prior[1]), euclidean(param_matrix_std, prior[2]), mean(prior[1]), mean(param_matrix_mean), mean(prior[2]), mean(param_matrix_std)]], ',')

	new_prior = (param_matrix_mean, param_matrix_std)

	# println("size of training data is: ",size(training_data))
	# println("The dimenstions of the new_pool and param_matrix during AL step no. $al_step are:", size(new_pool), " & ", size(param_matrix))
	return new_pool, new_prior, independent_param_matrix, training_data
end