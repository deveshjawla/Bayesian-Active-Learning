function running_active_learning(n_acq_steps, pool, n_input, n_output, test, experiment, pipeline_name, acquisition_size, acq_func, learning_algorithm)
    # n_acq_steps = 10#round(Int, total_pool_samples / acquisition_size, RoundUp)
    network_tuple, new_training_data = 0, 0
    last_elapsed = 0
    new_pool = 0
    for AL_iteration = 1:n_acq_steps
        if AL_iteration == 1
            new_pool, network_tuple, new_training_data, last_acc, last_elapsed = query(pool, new_training_data, n_input, n_output, network_tuple, AL_iteration, test, experiment, pipeline_name, acquisition_size, "Random", learning_algorithm)
            n_acq_steps = deepcopy(AL_iteration)
        elseif lastindex(new_pool[2]) >= acquisition_size
            new_pool, network_tuple, new_training_data, last_acc, last_elapsed = query(new_pool, new_training_data, n_input, n_output, network_tuple, AL_iteration, test, experiment, pipeline_name, acquisition_size, acq_func, learning_algorithm)
            n_acq_steps = deepcopy(AL_iteration)
            # elseif lastindex(new_pool[2]) <= acquisition_size && lastindex(new_pool[2]) > 0
            #     new_pool, network_tuple, new_training_data, last_acc, last_elapsed = query(new_pool, new_training_data, n_input, n_output, network_tuple, AL_iteration, test, experiment, pipeline_name, lastindex(new_pool[2]), acq_func, learning_algorithm)
            #     println("Trained on last few samples remaining in the Pool")
            #     n_acq_steps = deepcopy(AL_iteration)
        end
    end
    return n_acq_steps
end

function non_bayesian_get_sampled_indices(al_sampling, acq_size_, pool_size, pool_prediction_matrix)::Vector{Int}
    if al_sampling == "Random"
        sampled_indices = 1:acq_size_
        # elseif al_sampling == "Random"
        #     sampled_indices = random_acquisition(pool_size, acq_size_)
    elseif al_sampling == "PowerEntropy"
        entropy_scores = pool_prediction_matrix[3, :]
        sampled_indices = stochastic_acquisition(entropy_scores, acq_size_; acquisition_type="Power")
    elseif al_sampling == "StochasticEntropy"
        entropy_scores = pool_prediction_matrix[3, :]
        sampled_indices = stochastic_acquisition(entropy_scores, acq_size_; acquisition_type="Stochastic")
    elseif al_sampling == "Uncertainty"
        scores = pool_prediction_matrix[2, :]
        sampled_indices = top_k_acquisition_no_duplicates(scores, acq_size_; descending=true)
    elseif al_sampling == "Entropy"
        epistemic_uncertainties = pool_prediction_matrix[3, :]
        sampled_indices = top_k_acquisition_no_duplicates(epistemic_uncertainties, acq_size_; descending=true)
    end
    return sampled_indices
end


function collecting_stats_active_learning_experiments_classification(n_acq_steps, acq_func, experiment, pipeline_name, n_output)
    elapsed_stats = Array{Float32}(undef, 1, n_acq_steps)
    class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    cum_class_dist_data = Array{Int}(undef, n_output, n_acq_steps)
    performance_data = Array{Any}(undef, 7, n_acq_steps) #dims=(features, samples(i))
    cum_class_dist_ent = Array{Float32}(undef, 1, n_acq_steps)

    for al_step = 1:n_acq_steps

        elapsed = readdlm("./Experiments/$(experiment)/$(pipeline_name)/convergence_statistics/elapsed_$(al_step).csv", ',')
        elapsed_stats[:, al_step] = elapsed

        m = readdlm("./Experiments/$(experiment)/$(pipeline_name)/classification_performance/$(al_step).csv", ',')
        performance_data[1, al_step] = m[1, 2] #AcquisitionSize
        cd = readdlm("./Experiments/$(experiment)/$(pipeline_name)/query_batch_class_distributions/$(al_step).csv", ',')
        performance_data[2, al_step] = cd[1, 2] #ClassDistEntropy
        performance_data[3, al_step] = m[2, 2] #Accuracy Score
        performance_data[4, al_step] = m[3, 2] #F1 
        performance_data[5, al_step] = acq_func
        performance_data[6, al_step] = experiment
        performance_data[7, al_step] = al_step == 1 ? m[1, 2] : performance_data[7, al_step-1] + m[1, 2] #Cumulative Training Size
        for i in 1:n_output
            class_dist_data[i, al_step] = cd[i+1, 2]
            cum_class_dist_data[i, al_step] = al_step == 1 ? cd[i+1, 2] : cum_class_dist_data[i, al_step-1] + cd[i+1, 2]
        end
        cum_class_dist_ent[1, al_step] = normalized_entropy(cum_class_dist_data[:, al_step] ./ sum(cum_class_dist_data[:, al_step]), n_output)
    end

    kpi = vcat(performance_data, class_dist_data, cum_class_dist_data, cum_class_dist_ent, elapsed_stats)
    writedlm("./Experiments/$(experiment)/$(pipeline_name)/kpi.csv", kpi, ',')
    return kpi
end