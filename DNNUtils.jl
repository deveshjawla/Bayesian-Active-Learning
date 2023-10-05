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
            model = reconstruct(theta)
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
function pred_analyzer_multiclass(reconstruct, test_xs::Array{Float32,2}, params_set::Array{Float32,2})::Array{Float32,2}
    n_samples = size(test_xs)[2]
    ensemble_size = size(params_set)[1]
    pred_matrix = Array{Float32}(undef, 2, n_samples)
    for i = 1:n_samples
        predictions = []
        for j = 1:ensemble_size
            model = reconstruct(params_set[j, :])
            # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
            ŷ = model(test_xs[:, i])
            # ŷ = reconstruct(test_xs[:,i], params_set[j,:])#for the one with destructure
            # ŷ = reconstruct(test_xs[:,i], params_set[j,:], network_shape)#for the one with unpack
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
Returns a matrix of size { 2 (mean, std), n_samples }
"""
function pred_regression(reconstruct, test_xs::Array{Float32,2}, params_set::Array{Float32,2})::Array{Float32,2}
    n_samples = size(test_xs)[2]
    ensemble_size = size(params_set)[1]
    pred_matrix = Array{Float32}(undef, 2, n_samples)
    for i = 1:n_samples
        predictions = []
        for j = 1:ensemble_size
            model = reconstruct(params_set[j, :])
            ŷ = model(test_xs[:, i])
            append!(predictions, ŷ)
        end
		mean_ = mean(predictions)
		std_ = std(predictions)
        pred_matrix[:, i] = [mean_, std_]
    end
    return pred_matrix
end

"""
Returns a matrix of dims (n_output, ensemble_size, n_samples)
"""
function pool_predictions(reconstruct, test_xs::Array{Float32,2}, params_set::Array{Float32,2}, n_output::Int)::Array{Float32,3}
    n_samples = size(test_xs)[2]
    ensemble_size = size(params_set)[1]
    pred_matrix = Array{Float32}(undef, n_output, ensemble_size, n_samples)

    for i = 1:n_samples, j = 1:ensemble_size
        model = reconstruct(params_set[j, :])
        # # make probabilistic by inserting bernoulli distributions, we can make each prediction as probabilistic and then average out the predictions to give us the final predictions_mean and std
        ŷ = model(test_xs[:, i])
        pred_matrix[:, j, i] = ŷ
        # if i==100 && j==100
        # 	println(ŷ)
        # end
    end
    # println(pred_matrix[:,100,100])
    return pred_matrix
end

using Distributed

# # instantiate and precompile environment in all processes
# @everywhere begin
#     using Pkg
#     Pkg.activate(@__DIR__)
#     Pkg.instantiate()
#     Pkg.precompile()
# end

@everywhere begin
    # load dependencies
    using ProgressMeter
    using CSV


    multiclass_dnn_accuracy(x, y) = mean(onecold(softmax(model(x))) .== onecold(y))
    using Flux

    function network_training(n_epochs, input_size, output_size, train_loader; lr=1e-3, dropout_rate=0.2)::Vector{Float32}
        # model = Chain(
        #     Dense(input_size => input_size, mish; init=Flux.glorot_uniform()),
        #     Dropout(dropout_rate),
        #     Dense(input_size => input_size, mish; init=Flux.glorot_uniform()),
        #     Dropout(dropout_rate),
        #     Dense(input_size => output_size; init=Flux.glorot_uniform()),
        # )

		model =	Chain(
            Parallel(vcat, Dense(input_size => input_size, identity; init=Flux.glorot_uniform()), Dense(input_size => input_size, mish; init=Flux.glorot_uniform()), Dense(input_size => input_size, tanh; init=Flux.glorot_uniform()), Dense(input_size => input_size, sin; init=Flux.glorot_uniform())),
			# Dropout(dropout_rate),
			Parallel(vcat, Dense(4*input_size => input_size, identity; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, mish; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, tanh; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, sin; init=Flux.glorot_uniform())),
			# Dropout(dropout_rate),
			Parallel(vcat, Dense(4*input_size => input_size, identity; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, mish; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, tanh; init=Flux.glorot_uniform()), Dense(4*input_size => input_size, sin; init=Flux.glorot_uniform())),
			# Dropout(dropout_rate),
			Dense(4*input_size => output_size),
			softmax
			)
		opt = Adam(lr)
        opt_state = Flux.setup(opt, model)

        # @info("Beginning training loop...")
        least_loss = Inf32
        last_improvement = 0
        optim_params = 0

        for epoch_idx in 1:n_epochs
            # global best_acc, last_improvement
            loss = 0.0
            for (x, y) in train_loader
                # Compute the loss and the gradients:
                l, gs = Flux.withgradient(m -> Flux.logitcrossentropy(m(x), y), model)

                if !isfinite(l)
                    # @warn "loss is $l" epoch_idx
                    continue
                end
                begin
                    # Update the model parameters (and the Adam momenta):
                    Flux.update!(opt_state, model, gs[1])
                    # Accumulate the mean loss, just for logging:
                    loss += l / length(train_loader)
                end
            end

            if mod(epoch_idx, 2) == 1
                # Report on train and test, only every 2nd epoch_idx:
                # @info "After epoch_idx = $epoch_idx" loss
            end

            # If this is the minimum loss we've seen so far, save the model out
            if abs(loss) < abs(least_loss)
                # @info(" -> New minimum loss! Saving model weights")
                optim_params, re = Flux.destructure(model)
                least_loss = loss
                last_improvement = epoch_idx
            end

            # If we haven't seen improvement in 5 epochs, drop our learning rate:
            if epoch_idx - last_improvement >= 10 && opt_state.layers[1].layers[1].weight.rule.eta > 1e-6
                new_eta = opt_state.layers[1].layers[1].weight.rule.eta / 10.0
                # @warn(" -> Haven't improved in a while, dropping learning rate to $(new_eta)!")
                Flux.adjust!(opt_state; eta=new_eta)
                # After dropping learning rate, give it a few epochs to improve
                last_improvement = epoch_idx
            end

            if epoch_idx - last_improvement >= 30
                # @warn(" -> We're calling this converged.")
                break
            end
        end

        return optim_params
    end

end

using SharedArrays
function parallel_network_training(n_networks, nparameters, n_epochs, input_size, output_size, train_loader)::Matrix{Float32}
    param_matrices_accumulated = SharedMatrix{Float32}(n_networks, nparameters)
    pmap(1:n_networks) do i #@showprogress
        network_weights = network_training(n_epochs, input_size, output_size, train_loader)
        param_matrices_accumulated[i, :] = network_weights
    end
	return convert(Matrix{Float32}, param_matrices_accumulated)
end

function ensemble_training(num_params::Int, input_size::Int, output_size::Int, acq_size::Int, training_data::Tuple{Array{Float32,2},Array{Int,2}}, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String; n_epochs=1000)::Tuple{Array{Float32,2},Float32}
    train_x, train_y = training_data

    train_y = Flux.onehotbatch(vec(train_y), 1:output_size)
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # println(eltype(train_x), eltype(train_y))
    train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)

    burn_in = 0#Int(0.6 * nsteps)
    n_indep_samples = Int((nsteps - burn_in) / 10)
    ensemble_size = n_chains * n_indep_samples

    chain_timed = @timed parallel_network_training(ensemble_size, num_params, n_epochs, input_size, output_size, train_loader)

    param_matrices_accumulated = convert(Array{Float32,2}, chain_timed.value)
    elapsed = Float32(chain_timed.time)

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    return param_matrices_accumulated, elapsed
end