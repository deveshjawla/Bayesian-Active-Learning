"""
Returns new_pool_tuple, independent_param_matrix, independent_noise_x, independent_noise_y, training_data, acc, elapsed, param_matrix_mean
"""
function bnn_query(prior::Tuple, pool::Tuple, previous_training_data, n_input::Int, n_output::Int, categorical_indices_list, param_matrix, noise_x_set, noise_y_set, al_step::Int, test_data, experiment::String, pipeline_name::String, acq_size_::Int, nsteps::Int, n_chains::Int, al_sampling::String, mcmc_init_params, temperature, prior_informativeness, likelihood_name, learning_algorithm, noise_x, noise_y)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Array{Float32,2},Matrix{Float32},Matrix{Float32},Array{Float32,2},Float32,Float32,Vector{Float32}}

    new_pool_tuple, independent_param_matrix, independent_noise_x, independent_noise_y, acc, elapsed, param_matrix_mean = nothing, nothing, nothing, nothing, nothing, nothing, nothing

    println("$(al_sampling) with query no. ", al_step)
    # sigma, num_params = prior
    pool_x, pool_y = pool
    pool_size = lastindex(pool_y)
    sampled_indices = 0
    pool_prediction_matrix = 0
    if typeof(param_matrix) == Array{Float32,2}
        # pool_prediction_matrix = pred_analyzer_multiclass(pool_x, categorical_indices_list,  param_matrix; noise_set_x=noise_x_set)
        if noise_x && !noise_y
            pool_prediction_matrix = pred_analyzer_multiclass(pool_x, categorical_indices_list, param_matrix; noise_set_x=noise_x_set)
        elseif noise_y && !noise_x
            pool_prediction_matrix = pred_analyzer_multiclass(pool_x, categorical_indices_list, param_matrix; noise_set_y=noise_y_set)
        elseif noise_y && noise_x
            pool_prediction_matrix = pred_analyzer_multiclass(pool_x, categorical_indices_list, param_matrix; noise_set_x=noise_x_set, noise_set_y=noise_y_set)
        else
            pool_prediction_matrix = pred_analyzer_multiclass(pool_x, categorical_indices_list, param_matrix)
        end
    end
    sampled_indices = get_sampled_indices(al_sampling, acq_size_, pool_size, pool_prediction_matrix)

    pool = vcat(pool_x, pool_y)
    new_acq_size_ = lastindex(sampled_indices)
    new_training_data = pool[:, sampled_indices]
    new_pool = pool[:, Not(sampled_indices)]
    # if n_output == 2 && class_balancing == "BalancedBinaryAcquisition"
    #     new_training_data, leftovers = balance_binary_data(new_training_data, positive_class_label=1, negative_class_label=2)
    #     if leftovers !== nothing
    #         new_pool = hcat(new_pool, leftovers)
    #         new_acq_size_ = new_acq_size_ - size(leftovers, 2)
    #     end
    # end

    # if new_acq_size_ == 0
    #     @info "Last Query Returned Zero Samples, therefore Random Sampling!"
    #     sampled_indices = random_acquisition(pool_size, acq_size_)
    #     acq_size_ = lastindex(sampled_indices)
    #     new_training_data = pool[:, sampled_indices]
    #     new_pool = pool[:, Not(sampled_indices)]
    #     # if n_output == 2 && class_balancing == "BalancedBinaryAcquisition"
    #     #     new_training_data, leftovers = balance_binary_data(new_training_data, positive_class_label=1, negative_class_label=2)
    #     #     if leftovers !== nothing
    #     #         new_pool = hcat(new_pool, leftovers)
    #     #         acq_size_ = acq_size_ - size(leftovers, 2)
    #     #     end
    #     # end
    # else
    acq_size_ = new_acq_size_
    # end

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

    sample_weights = nothing
    if n_output > 1
        #calculate the weights of the samples
        balance_of_training_data = countmap(Int.(training_data_y))

        sample_weights = similar(training_data_y, Float32)

        nos_training = lastindex(training_data_y)

        for i = 1:nos_training
            sample_weights[i] = nos_training / balance_of_training_data[training_data_y[i]]
        end
        sample_weights ./= n_output
    end

    # @info "Sample weights type" typeof(sample_weights)

    # println("The acquired Batch has the follwing class distribution: $balance_of_acquired_batch")
    if n_output == 1
        training_data_xy = (training_data_x, Float32.(permutedims(training_data_y)))
    else
        training_data_xy = (training_data_x, Int.(permutedims(training_data_y)))
    end
    # println("The dimenstions of the training data during AL step no. $al_step are:", size(training_data_x))

    #Training on Acquired Samples and logging classification_performance
    if learning_algorithm == "VI"
        independent_param_matrix, independent_noise_x, elapsed = vi_inference(prior, training_data_xy, al_step, experiment, pipeline_name, temperature, sample_weights, likelihood_name)
    elseif learning_algorithm == "MCMC"
        # independent_param_matrix, independent_noise_x, elapsed = mcmc_inference(prior, training_data_xy, n_input, nsteps, n_chains, al_step, experiment, pipeline_name, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness, noise_x, noise_y)

        independent_param_matrix, independent_noise_x, independent_noise_y, elapsed = mcmc_inference(prior, training_data_xy, n_input, n_output, categorical_indices_list, nsteps, n_chains, al_step, experiment, pipeline_name, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness, noise_x, noise_y)
    end
    test_x, test_y = test_data

    # activations_weighted_sums(test_x[:,1], independent_param_matrix, "./Experiments/$(experiment)/$(pipeline_name)/", 3) # 3 is the number of layers
    # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$al_step.csv", predictions, ',')
    # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/$(al_step)_map.csv", predictions_map, ',')
    # println("Checking if dimensions of test_y and ŷ_test are same", size(test_y), size(ŷ_test))
    if n_output == 1
        predictions_mean, predictions_std = pred_analyzer_regression(test_x, independent_param_matrix; noise_set_x=independent_noise_x)

        ŷ_test = permutedims(predictions_mean)

        mse, mae = performance_stats_regression(test_y, ŷ_test)
        acc = mse
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "MSE", "MAE"] [acq_size_, mse, mae]], ',')
    else
        # predictions = pred_analyzer_multiclass(pool_x, categorical_indices_list,  independent_param_matrix; noise_set_x=independent_noise_x)
        if noise_x && !noise_y
            predictions = pred_analyzer_multiclass(test_x, categorical_indices_list, independent_param_matrix, noise_set_x=independent_noise_x)
        elseif noise_y && !noise_x
            predictions = pred_analyzer_multiclass(test_x, categorical_indices_list, independent_param_matrix, noise_set_y=independent_noise_y)
        elseif noise_y && noise_x
            predictions = pred_analyzer_multiclass(test_x, categorical_indices_list, independent_param_matrix, noise_set_x=independent_noise_x, noise_set_y=independent_noise_y)
        else
            predictions = pred_analyzer_multiclass(test_x, categorical_indices_list, independent_param_matrix)
        end

        ŷ_test = predictions[1, :]

        # cols = [:Confidence, :StdDeviationConfidence, :Aleatoric, :Epistemic, :TotalUncertainty]
        # M = cor(predictions[2:6, :], dims=2)
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/corr_matrix_$al_step.csv", M, ',')
        # plotter(M, cols)

        acc, f1 = performance_stats_multiclass(test_y, ŷ_test, n_output)
        writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$al_step.csv", [["Acquisition Size", "Balanced Accuracy", "MacroF1Score"] [acq_size_, acc, f1]], ',')

        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step)_map.csv", [["Acquisition Size", "Accuracy"] [acq_size_, acc_map]], ',')
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
    new_pool_tuple = (new_pool[1:n_input, :], permutedims(new_pool[end, :]))

    return new_pool_tuple, independent_param_matrix, independent_noise_x, independent_noise_y, training_data, acc, elapsed, param_matrix_mean
end