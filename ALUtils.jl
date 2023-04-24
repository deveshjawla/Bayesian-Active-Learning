using Distances


function diversity_sampling(prior, pool, input_size, n_output, al_step, test_data, name_exp; acquisition_size = 50, nsteps = 1000, n_chains = 5)
	println("diversity_sampling with AL step no. ", al_step)
	pool_x, pool_y = pool
	#Using Diversity Sampling we acquire the initial set
	pool = vcat(pool_x, pool_y)
	pool_size = size(pool)[2]
	# acquisition_size = round(Int, 0.2*pool_size)
	random_query = initial_random_acquisition(pool_size, acquisition_size)
	new_pool = pool[:, Not(random_query)]

	#Training on Acquired Samples and logging classification_performance
	new_training_data = pool[:, random_query]
	new_training_data_y = new_training_data[end, :]
	balance_of_acquired_batch = countmap(new_training_data_y)

	training_data = new_training_data

	training_data_x, training_data_y = training_data[1:input_size, :], training_data[end, :]
	# println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
	training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
	# println("The dimenstions of the training data during AL step no. $al_step is:", size(training_data_x))
	independent_param_matrix = bayesian_inference(prior, training_data_xy, nsteps, n_chains, al_step, name_exp)
	
	test_x, test_y = test_data
	predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", predictions, ',')
	ŷ_test = permutedims(Int.(predictions[1,:]))
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
	else
		acc = accuracy_multiclass(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]], ',')
	end
	#Turning the posterior obtained after training on new samples into the new prior for the next iteration
    param_matrix_mean = vec(mean(independent_param_matrix, dims=1))
    param_matrix_std = vec(std(independent_param_matrix, dims=1))
    # println("Euclidean distance between Prior and Posterior distribution's means is  ", euclidean(param_matrix_mean, prior[1]))
    # println("Euclidean distance between Prior and Posterior distribution's stds is ", euclidean(param_matrix_std, prior[2]))
	# println("Means of Prior and Posterior distribution's Means are  ", mean(prior[1]), " & ", mean(param_matrix_mean))	
	# println("Means of Prior and Posterior distribution's stds are  ", mean(prior[2]), " & ", mean(param_matrix_std))

	writedlm("./experiments/$(name_exp)/log_distribution_changes/$al_step.csv", [["Euclidean distance between Prior and Posterior distribution's means is  ","Euclidean distance between Prior and Posterior distribution's stds is ", "Means of Prior", "and Posterior distribution's Means are  ", "Means of Prior", "and Posterior distribution's stds are  "] [euclidean(param_matrix_mean, prior[1]), euclidean(param_matrix_std, prior[2]), mean(prior[1]), mean(param_matrix_mean), mean(prior[2]), mean(param_matrix_std)]], ',')

	new_prior = (param_matrix_mean, param_matrix_std)

	return new_pool, new_prior, independent_param_matrix, training_data
end

"""
Returns new_pool, new_prior, independent_param_matrix
"""
function uncertainty_sampling(prior, pool, previous_training_data, input_size, n_output, param_matrix, al_step, test_data, name_exp; acquisition_size = 10, nsteps = 1000, n_chains = 5)
	println("uncertainty_sampling with AL step no. ", al_step)
	pool_x, pool_y = pool[1:input_size, :], pool[end, :]
	#Scoring the pool and acquiring new samples
	pool_prediction_matrix = pool_predictions(pool_x, param_matrix, n_output)
	pool_scores = mapslices(x->bald(x,n_output), pool_prediction_matrix, dims=[1,2])
	bald_scores = map(x->x[2], pool_scores[1,1,:])
	# var_ratio_scores = 1 .- pŷ_test

	# random_query = random_acquisition(bald_scores, acquisition_size)
	# softmax_entropy = softmax_acquisition(entropy_scores, acquisition_size)
	power_bald = power_acquisition(bald_scores, acquisition_size)

	new_pool = pool[:, Not(power_bald)]

	#Training on Acquired Samples and logging classification_performance
	new_training_data = pool[:, power_bald]
	new_training_data_y = new_training_data[end, :]
	balance_of_acquired_batch = countmap(new_training_data_y)

	training_data = hcat(previous_training_data, new_training_data)

	training_data_x, training_data_y = training_data[1:input_size, :], training_data[end, :]
	# println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
	training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
	# println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))
	independent_param_matrix = bayesian_inference(prior, training_data_xy, nsteps, n_chains, al_step, name_exp)
	
	test_x, test_y = test_data
	predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", predictions, ',')
	ŷ_test = permutedims(Int.(predictions[1,:]))
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]])
	else
		acc = accuracy_multiclass(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size,balance_of_acquired_batch, acc]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]])
	end

	#Turning the posterior obtained after training on new samples into the new prior for the next iteration
    param_matrix_mean = vec(mean(independent_param_matrix, dims=1))
    param_matrix_std = vec(std(independent_param_matrix, dims=1))
    # println("Euclidean distance between Prior and Posterior distribution's means is  ", euclidean(param_matrix_mean, prior[1]))
    # println("Euclidean distance between Prior and Posterior distribution's stds is ", euclidean(param_matrix_std, prior[2]))
	# println("Means of Prior and Posterior distribution's Means are  ", mean(prior[1]), " & ", mean(param_matrix_mean))	
	# println("Means of Prior and Posterior distribution's stds are  ", mean(prior[2]), " & ", mean(param_matrix_std))

	writedlm("./experiments/$(name_exp)/log_distribution_changes/$al_step.csv", [["Euclidean distance between Prior and Posterior distribution's means is  ","Euclidean distance between Prior and Posterior distribution's stds is ", "Means of Prior", "and Posterior distribution's Means are  ", "Means of Prior", "and Posterior distribution's stds are  "] [euclidean(param_matrix_mean, prior[1]), euclidean(param_matrix_std, prior[2]), mean(prior[1]), mean(param_matrix_mean), mean(prior[2]), mean(param_matrix_std)]], ',')

	new_prior = (param_matrix_mean, param_matrix_std)

	println("size of training data is: ",size(training_data))

	# println("The dimenstions of the new_pool and param_matrix during AL step no. $al_step are:", size(new_pool), " & ", size(param_matrix))
	return new_pool, new_prior, independent_param_matrix, training_data
end


"""
Returns new_pool, new_prior, independent_param_matrix
"""
function random_sampling(prior, pool, previous_training_data, input_size, n_output, param_matrix, al_step, test_data, name_exp; acquisition_size = 10, nsteps = 1000, n_chains = 5)
	println("random_sampling with AL step no. ", al_step)

	pool_x, pool_y = pool[1:input_size, :], pool[end, :]
	pool_size = lastindex(pool_y)
	random_query = initial_random_acquisition(pool_size, acquisition_size)
	new_pool = pool[:, Not(random_query)]

	#Training on Acquired Samples and logging classification_performance
	new_training_data = pool[:, random_query]
	new_training_data_y = new_training_data[end, :]
	balance_of_acquired_batch = countmap(new_training_data_y)

	training_data = hcat(previous_training_data, new_training_data)

	training_data_x, training_data_y = training_data[1:input_size, :], training_data[end, :]
	# println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
	training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
	# println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))
	independent_param_matrix = bayesian_inference(prior, training_data_xy, nsteps, n_chains, al_step, name_exp)
	
	test_x, test_y = test_data
	predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", predictions, ',')
	ŷ_test = permutedims(Int.(predictions[1,:]))
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]])
	else
		acc = accuracy_multiclass(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size,balance_of_acquired_batch, acc]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]])
	end

	#Turning the posterior obtained after training on new samples into the new prior for the next iteration
    param_matrix_mean = vec(mean(independent_param_matrix, dims=1))
    param_matrix_std = vec(std(independent_param_matrix, dims=1))
    # println("Euclidean distance between Prior and Posterior distribution's means is  ", euclidean(param_matrix_mean, prior[1]))
    # println("Euclidean distance between Prior and Posterior distribution's stds is ", euclidean(param_matrix_std, prior[2]))
	# println("Means of Prior and Posterior distribution's Means are  ", mean(prior[1]), " & ", mean(param_matrix_mean))	
	# println("Means of Prior and Posterior distribution's stds are  ", mean(prior[2]), " & ", mean(param_matrix_std))

	writedlm("./experiments/$(name_exp)/log_distribution_changes/$al_step.csv", [["Euclidean distance between Prior and Posterior distribution's means is  ","Euclidean distance between Prior and Posterior distribution's stds is ", "Means of Prior", "and Posterior distribution's Means are  ", "Means of Prior", "and Posterior distribution's stds are  "] [euclidean(param_matrix_mean, prior[1]), euclidean(param_matrix_std, prior[2]), mean(prior[1]), mean(param_matrix_mean), mean(prior[2]), mean(param_matrix_std)]], ',')

	new_prior = (param_matrix_mean, param_matrix_std)

	# println("The dimenstions of the new_pool and param_matrix during AL step no. $al_step are:", size(new_pool), " & ", size(param_matrix))
	return new_pool, new_prior, independent_param_matrix, training_data
end