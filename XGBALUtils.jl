
function diversity_sampling(pool, input_size, n_output, al_step, test_data, name_exp; acquisition_size = 50, nsteps = 1000)
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

	training_data_x, training_data_y = copy(transpose(training_data[1:input_size, :])), vec(copy(transpose(training_data[end, :])))

	xgb = xgboost((training_data_x, training_data_y), num_round=nsteps, max_depth=6, objective="multi:softmax", num_class = n_output)
	xgb_random = xgboost((training_data_x, training_data_y), num_round=nsteps, max_depth=6, objective="multi:softmax", num_class = n_output)
	
	test_x, test_y = copy(transpose(test_data[1])), vec(copy(transpose(test_data[2])))
	ŷ_test = XGBoost.predict(xgb, test_x)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", ŷ_test, ',')
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
		writedlm("./experiments/random_sampling_$(acquisition_size)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
	else
		acc = mean(test_y.==ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]], ',')
		writedlm("./experiments/random_sampling_$(acquisition_size)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]], ',')
	end

	return new_pool, xgb, xgb_random, training_data
end

function uncertainty_sampling(xgb::Booster, pool, previous_training_data, input_size, n_output, al_step, test_data, name_exp; acquisition_size = 10, nsteps = 1000)
	println("uncertainty_sampling with AL step no. ", al_step)
	pool_x, pool_y = pool[1:input_size, :], pool[end, :]
	#Scoring the pool and acquiring new samples
	pool_x= copy(transpose(pool_x))
	pool_prediction_matrix = XGBoost.predict(xgb, pool_x, margin =true)
	entropy_scores = mapslices(x->normalized_entropy(softmax(x),n_output), pool_prediction_matrix, dims=2)
	# var_ratio_scores = 1 .- pŷ_test
	# random_query = random_acquisition(entropy_scores, acquisition_size)
	# softmax_entropy = softmax_acquisition(entropy_scores, acquisition_size)
	power_bald = power_acquisition(vec(entropy_scores), acquisition_size)

	new_pool = pool[:, Not(power_bald)]

	#Training on Acquired Samples and logging classification_performance
	new_training_data = pool[:, power_bald]
	new_training_data_y = new_training_data[end, :]
	balance_of_acquired_batch = countmap(new_training_data_y)

	training_data = hcat(previous_training_data, new_training_data)

	training_data_x, training_data_y = copy(transpose(training_data[1:input_size, :])), vec(copy(transpose(training_data[end, :])))

	update!(xgb, (training_data_x, training_data_y), num_round=nsteps)
	
	test_x, test_y = copy(transpose(test_data[1])), vec(copy(transpose(test_data[2])))
	ŷ_test = XGBoost.predict(xgb, test_x)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", ŷ_test, ',')
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
	else
		acc = mean(test_y.==ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size,balance_of_acquired_batch, acc]], ',')
	end

	println("size of training data is: ",size(training_data))

	return new_pool, xgb, training_data
end

function random_sampling(xgb::Booster, pool, previous_training_data, input_size, n_output, al_step, test_data, name_exp; acquisition_size = 10, nsteps = 1000)
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

	training_data_x, training_data_y = copy(transpose(training_data[1:input_size, :])), vec(copy(transpose(training_data[end, :])))
	
	update!(xgb, (training_data_x, training_data_y), num_round=nsteps)
	
	test_x, test_y = copy(transpose(test_data[1])), vec(copy(transpose(test_data[2])))
	ŷ_test = XGBoost.predict(xgb, test_x)
    writedlm("./experiments/$(name_exp)/predictions/$al_step.csv", ŷ_test, ',')
	# println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
	# pŷ_test = predictions[:,2]
	if n_output == 2
		acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution", "Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acquisition_size, balance_of_acquired_batch, acc, mcc, f1, fpr, prec, recall, threat, cm]])
	else
		acc = mean(test_y.==ŷ_test)
		writedlm("./experiments/$(name_exp)/classification_performance/$al_step.csv", [["Acquisition Size","Acquired Batch class distribution", "Accuracy"] [acquisition_size,balance_of_acquired_batch, acc]], ',')
		# println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acquisition_size, balance_of_acquired_batch, acc]])
	end

	# println("The dimenstions of the new_pool and param_matrix during AL step no. $al_step are:", size(new_pool), " & ", size(param_matrix))
	return new_pool, xgb, training_data
end