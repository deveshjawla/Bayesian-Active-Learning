"""
Returns new_pool, new_prior, learned_weights, training_data
"""
function query(pool::Tuple, previous_training_data, n_input::Int, n_output::Int, old_network_tuple, al_step::Int, test_data, experiment::String, pipeline_name::String, acq_size_::Int, al_sampling::String, learning_algorithm)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Tuple{Any,Any},Array{Float32,2},Float32,Float32}
    println("$(al_sampling) with query no. ", al_step)
    pool_x, pool_y = pool #pool_y is a onehotbatch
    pool_size = lastindex(pool_y)
    # pool_y = mapslices(argmax, pool_y, dims=1)
    sampled_indices = 0
    pool_prediction_matrix = 0

    if al_step != 1
        old_weights, nn_reconstruct = old_network_tuple
        if learning_algorithm == "Evidential"
            old_network = nn_reconstruct(old_weights)
            α̂ = old_network(pool_x)
            ŷ = α̂ ./ sum(α̂, dims=1)
            ŷ_label = mapslices(argmax, ŷ, dims=1)
            u = edlc_uncertainty(α̂)
            entropies_edlc = mapslices(x -> normalized_entropy(x, n_output), ŷ, dims=1)
            pool_prediction_matrix = vcat(ŷ_label, u, entropies_edlc)
        elseif learning_algorithm == "LaplaceApprox"
            predictions_pool_x = LaplaceRedux.predict(nn_reconstruct, pool_x, link_approx=:probit)
            pool_prediction_matrix = pred_analyzer_multiclass(predictions_pool_x, n_output)
        elseif learning_algorithm == "RBF"
            predictions_pool_x = testRbf(old_weights[1], old_weights[2], nn_reconstruct[1], nn_reconstruct[2], permutedims(pool_x))
            pool_prediction_matrix = pred_analyzer_multiclass(predictions_pool_x, n_output)
        end
    end

    sampled_indices = non_bayesian_get_sampled_indices(al_sampling, acq_size_, pool_size, pool_prediction_matrix)

    pool = vcat(pool_x, pool_y)
    acq_size_ = lastindex(sampled_indices)
    @info "Acquisition size is $(acq_size_)"
    new_training_data = pool[:, sampled_indices]
    new_pool = pool[:, Not(sampled_indices)]

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

    class_dist_ent = normalized_entropy(class_dist[:, 2] ./ sum(class_dist[:, 2]), n_output)

    if al_step == 1
        training_data = copy(new_training_data)
    else
        if experiment == "NonCumulative"
            training_data = copy(new_training_data)
        else
            training_data = hcat(previous_training_data, new_training_data)
        end
    end

    training_data_x, training_data_y = training_data[1:n_input, :], training_data[end, :]

    #calculate the weights of the samples
    balance_of_training_data = countmap(Int.(training_data_y))

    sample_weights = similar(training_data_y, Float32)

    nos_training = lastindex(training_data_y)

    for i = 1:nos_training
        sample_weights[i] = nos_training / balance_of_training_data[training_data_y[i]]
    end
    sample_weights ./= n_output

    # @info "Sample weights type" typeof(sample_weights)

    # println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
    if n_output == 1
        training_data_xy = (training_data_x, Float32.(permutedims(training_data_y)))
    else
        training_data_xy = (training_data_x, Flux.onehotbatch(training_data_y, 1:n_output))
    end
    println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))

    #Training on Acquired Samples and logging classification_performance
    if learning_algorithm == "RBF"
        timed_rbf = @timed trainRbf(permutedims(training_data_x), vec(Int.(training_data_y)), round(Int, (acq_size_ / (n_output^2)), RoundUp), true, n_output)
        rbf_tuples = timed_rbf.value
        elapsed = timed_rbf.time
        learned_weights = (rbf_tuples[1], rbf_tuples[2])
        nn_reconstruct = (rbf_tuples[3], rbf_tuples[4])
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/elapsed_$(al_step).csv", [elapsed], ',')
        # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/optim_theta_$(al_step).csv", optim_theta, ',')
    else
        learned_weights, nn_reconstruct, elapsed = inference(training_data_xy, al_step, experiment, pipeline_name, learning_algorithm)
    end
    test_x, test_y = test_data

    if learning_algorithm == "Evidential"
        new_network = nn_reconstruct(learned_weights)
        α̂ = new_network(test_x)
        ŷ = α̂ ./ sum(α̂, dims=1)
        ŷ_label = mapslices(argmax, ŷ, dims=1)
        u = edlc_uncertainty(α̂)
        entropies_edlc = mapslices(x -> normalized_entropy(x, n_output), ŷ, dims=1)
        predictions_matrix = vcat(ŷ_label, u, entropies_edlc)
    elseif learning_algorithm == "LaplaceApprox"
        @info "Starting Predictions on Test"
        predictions_test_x = LaplaceRedux.predict(nn_reconstruct, test_x, link_approx=:probit)
        predictions_matrix = pred_analyzer_multiclass(predictions_test_x, n_output)
        @info "Finished Predictions on Test"
    elseif learning_algorithm == "RBF"
        predictions_test_x = testRbf(learned_weights[1], learned_weights[2], nn_reconstruct[1], nn_reconstruct[2], permutedims(test_x))
        predictions_matrix = pred_analyzer_multiclass(predictions_test_x, n_output)
    end

    # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions_matrix/$al_step.csv", predictions_matrix, ',')
    # println("Checking if dimensions of test_y and ŷ_test are same", size(test_y), size(ŷ_test))

    if n_output == 1
        ŷ_test = predictions_matrix[1, :]

        mse, mae = performance_stats_regression(test_y, ŷ_test)
        acc = mse
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "MSE", "MAE"] [acq_size_, mse, mae]], ',')
    else
        ŷ_test = predictions_matrix[1, :]

        acc, f1 = performance_stats_multiclass(test_y, ŷ_test, n_output)
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Balanced Accuracy", "MacroF1Score"] [acq_size_, acc, f1]], ',')

        writedlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$al_step.csv", ["ClassDistEntropy" class_dist_ent; class_dist], ',')
        # println([["Acquisition Size","Acquired Batch class distribution","Accuracy"] [acq_size_, balance_of_acquired_batch, acc]])
    end

    new_network_tuple = (learned_weights, nn_reconstruct)

    # println("size of training data is: ",size(training_data))
    # println("The dimenstions of the new_pool and old_network_tuple during AL step no. $al_step are:", size(new_pool), " & ", size(old_network_tuple))
    new_pool_tuple = (new_pool[1:n_input, :], permutedims(new_pool[end, :]))
    return new_pool_tuple, new_network_tuple, training_data, acc, elapsed
end