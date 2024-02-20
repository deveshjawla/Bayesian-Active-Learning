"""
Returns new_pool, new_xgb, training_data
"""
function xgb_query(xgb, pool, previous_training_data, input_size, n_output, al_step, test_data, experiment_name, pipeline_name, acq_size_, nsteps, al_sampling)

    println("$(al_sampling) with query step no. ", al_step)

    pool_x, pool_y = pool
    pool = vcat(pool_x, pool_y)
    pool_size = lastindex(pool_y)
    sampled_indices = 0
    if al_sampling == "Initial"
        sampled_indices = 1:acq_size_
    elseif al_sampling == "Random"
        sampled_indices = initial_random_acquisition(pool_size, acq_size_)
    elseif al_sampling == "PowerEntropy"
        #Scoring the pool and acquiring new samples
        pool_x = copy(permutedims(pool_x))
        pool_prediction_matrix = XGBoost.predict(xgb, pool_x, margin=true)
        entropy_scores = mapslices(x -> normalized_entropy(softmax(x), n_output), pool_prediction_matrix, dims=2)
        # var_ratio_scores = 1 .- pŷ_test
        # sampled_indices = random_acquisition(entropy_scores, acq_size_)
        sampled_indices = power_acquisition(vec(entropy_scores), acq_size_)
    elseif al_sampling == "Diversity"
        sampled_indices = initial_random_acquisition(pool_size, acq_size_)
    end
    #Using Diversity Sampling we acquire the initial set

    acq_size_ = lastindex(sampled_indices)
    new_training_data = pool[:, sampled_indices]
    new_pool = pool[:, Not(sampled_indices)]

    new_training_data_y = new_training_data[end, :]
    balance_of_acquired_batch = countmap(Int.(new_training_data_y))

    class_dist = Array{Int}(undef, n_output, 2)
    for i in 1:n_output
        try
            class_dist[i, 1] = i
            class_dist[i, 2] = balance_of_acquired_batch[i-1]
        catch
            class_dist[i, 1] = i
            class_dist[i, 2] = 0
        end
    end

    class_dist_ent = normalized_entropy(softmax(class_dist[:, 2]), n_output)

    if al_step == 1
        training_data = copy(new_training_data)
    else
        # training_data = copy(new_training_data)
        training_data = hcat(previous_training_data, new_training_data)
    end


    training_data_x, training_data_y = copy(permutedims(training_data[1:input_size, :])), vec(copy(permutedims(training_data[end, :])))

    #calculate the weights of the samples
    balance_of_training_data = countmap(Int.(training_data_y))
    sample_weights = similar(training_data_y, Float32)
    nos_training = lastindex(training_data_y)
    for i = 1:nos_training
        sample_weights[i] = nos_training / balance_of_training_data[training_data_y[i]]
    end

    #Training on Acquired Samples and logging classification_performance
    if al_step == 1
        xgb_timed = @timed xgboost((training_data_x, training_data_y), num_round=nsteps, max_depth=6, objective="multi:softmax", num_class=n_output)
        xgb = xgb_timed.value
        elapsed = xgb_timed.time
    else
        # xgb_timed = @timed update!(xgb, (training_data_x, training_data_y), num_round=nsteps)
        # _ = xgb_timed.value
        # elapsed = xgb_timed.time
		xgb_timed = @timed xgboost((training_data_x, training_data_y), num_round=nsteps, max_depth=6, objective="multi:softmax", num_class=n_output)
        xgb = xgb_timed.value
        elapsed = xgb_timed.time
    end

    test_x, test_y = copy(permutedims(test_data[1])), vec(copy(permutedims(test_data[2])))
    # ŷ_test = XGBoost.predict(xgb, test_x, margin=false)
    ŷ_test_prob = XGBoost.predict(xgb, test_x, margin=true)
    ŷ_test = mapslices(x -> argmax(softmax(x)), ŷ_test_prob, dims=2) .- 1
    pred_prob = mapslices(x -> maximum(softmax(x)), ŷ_test_prob, dims=2)
    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/predictions/$al_step.csv", [ŷ_test pred_prob], ',')
    # println("Checking if dimensions of test_y and ŷ_test are", size(test_y), size(ŷ_test))
    # pŷ_test = predictions[:,2]
    if n_output == 2
        acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Accuracy", "Elapsed", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acq_size_, acc, elapsed, mcc, f1, fpr, prec, recall, threat, cm]], ',')
        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
    else
        acc = accuracy_multiclass(test_y, ŷ_test)
        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Accuracy", "Elapsed"] [acq_size_, acc, elapsed]], ',')
        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
    end

    println("size of training data is: ", size(training_data))
    new_pool_tuple = (new_pool[1:input_size, :], permutedims(new_pool[end, :]))
    return new_pool_tuple, xgb, training_data, acc, elapsed
end



