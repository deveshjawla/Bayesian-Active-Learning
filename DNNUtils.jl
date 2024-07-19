
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

    function network_training(n_epochs, input_size, output_size, train_loader, sample_weights_loader; lr=1e-3, dropout_rate=0.2)::Vector{Float32}
        model = Chain(
            Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
			Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
			Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
            Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
            Dense(input_size => output_size; init=Flux.glorot_normal()),
			softmax
        )

        # model = Chain(
        #     Parallel(vcat, Dense(input_size => input_size, relu; init=Flux.glorot_normal()), Dense(input_size => input_size, relu; init=Flux.glorot_normal()), Dense(input_size => input_size, relu; init=Flux.glorot_normal()), Dense(input_size => input_size, relu; init=Flux.glorot_normal())),
        #     # Dropout(dropout_rate),
        #     Parallel(vcat, Dense(4 * input_size => input_size, relu; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, relu; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, relu; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, relu; init=Flux.glorot_normal())),
        #     # Dropout(dropout_rate),
        #     Dense(4 * input_size => output_size),
        #     softmax
        # )
        opt = Flux.Optimise.AdaBelief()
        opt_state = Flux.setup(opt, model)

        # @info("Beginning training loop...")
        least_loss = Inf32
        last_improvement = 0
        optim_params = 0

        for epoch_idx in 1:n_epochs
            # global best_acc, last_improvement
            loss = 0.0
            for ((x, y), sample_weights) in zip(train_loader, sample_weights_loader)
                # Compute the loss and the gradients:
                l, gs = Flux.withgradient(m -> logitcrossentropyweighted(m(x), y, sample_weights), model)

                if !isfinite(l)
                    @warn "loss is $l" epoch_idx
                    # continue
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

            # # If we haven't seen improvement in 5 epochs, drop our learning rate:
            # if epoch_idx - last_improvement >= 10 && opt_state.layers[1].weight.rule.eta > 1e-6
            #     new_eta = opt_state.layers[1].weight.rule.eta / 10.0
            #     # @warn(" -> Haven't improved in a while, dropping learning rate to $(new_eta)!")
            #     Flux.adjust!(opt_state; eta=new_eta)
            #     # After dropping learning rate, give it a few epochs to improve
            #     last_improvement = epoch_idx
            # end

            if epoch_idx - last_improvement >= 30
                # @warn(" -> We're calling this converged.")
                break
            end
        end

        return optim_params
    end

end

using SharedArrays
function parallel_network_training(n_networks, nparameters, n_epochs, input_size, output_size, train_loader, sample_weights_loader)::Matrix{Float32}
    param_matrices_accumulated = SharedMatrix{Float32}(n_networks, nparameters)
    @showprogress "Parallel Networks Training Progress" pmap(1:n_networks) do i #@showprogress
        network_weights = network_training(n_epochs, input_size, output_size, train_loader, sample_weights_loader)
        param_matrices_accumulated[i, :] = network_weights
    end
    return convert(Matrix{Float32}, param_matrices_accumulated)
end

function ensemble_training(num_params::Int, input_size::Int, output_size::Int, acq_size::Int, training_data::Tuple{Array{Float32,2},Array{Int,2}}, ensemble_size::Int, sample_weights; n_epochs=100)::Tuple{Array{Float32,2},Float32}
    train_x, train_y = training_data

    train_y = Flux.onehotbatch(vec(train_y), 1:output_size)
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # println(eltype(train_x), eltype(train_y))
    train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)
    sample_weights_loader = Flux.DataLoader(permutedims(sample_weights), batchsize=acq_size)

    chain_timed = @timed parallel_network_training(ensemble_size, num_params, n_epochs, input_size, output_size, train_loader, sample_weights_loader)

    param_matrices_accumulated = convert(Array{Float32,2}, chain_timed.value)
    elapsed = Float32(chain_timed.time)

    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    return param_matrices_accumulated, elapsed
end