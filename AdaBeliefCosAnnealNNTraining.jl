function logitcrossentropyweighted(ŷ::AbstractArray, y::AbstractArray, sample_weights::AbstractArray; dims=1)
    if size(ŷ) != size(y)
        error("logitcrossentropyweighted(ŷ, y), sizes of (ŷ, y) are not the same")
    end
    mean(.-sum(sample_weights .* (y .* logsoftmax(ŷ; dims=dims)); dims=dims))
end

function pred_analyzer_multiclass(preds::Matrix)::Array{Float32,2}
	pred_label = map(argmax, eachcol(preds))
    pred_prob = map(maximum, eachcol(preds))
    pred_plus_std = vcat(permutedims(pred_label), permutedims(1 .- pred_prob))
	return pred_plus_std
end

include("./MakeNNArch.jl")
using ParameterSchedulers
function network_training(nn_arch::String, input_size::Int, output_size::Int, n_epochs::Int; train_loader=nothing, sample_weights_loader=nothing, data=nothing, loss_function=false, lambda=0.0)::Tuple{Vector{Float32},Any}

    nn = make_nn_arch(nn_arch, input_size, output_size)

    opt = OptimiserChain(WeightDecay(lambda), AdaBelief())
    s = ParameterSchedulers.Stateful(CosAnneal(0.1, 1e-6, 100))
    opt_state = Flux.setup(opt, nn)

    # Train it
    least_loss = Inf32
    last_improvement = 0
    optim_params = 0
    re = 0

    trnlosses = zeros(n_epochs)
    for e in 1:n_epochs
        loss = 0.0
        Flux.adjust!(opt_state, ParameterSchedulers.next!(s))

        if !isnothing(train_loader) && nn_arch == "Evidential Classification"
            for (x, y) in train_loader
                # Compute the loss and the gradients:
                local l = 0.0
                l, grad = Flux.withgradient(m -> dirloss(y, m(x), e), nn)
                Flux.update!(opt_state, nn, grad[1])
                # Accumulate the mean loss, just for logging:
                loss += l / lastindex(train_loader)
            end
		elseif !isnothing(train_loader) && nn_arch != "Evidential Classification"
				for (x, y) in train_loader
					# Compute the loss and the gradients:
					local l = 0.0
					l, grad = Flux.withgradient(m -> loss_function(m(x), y), nn)
					Flux.update!(opt_state, nn, grad[1])
					# Accumulate the mean loss, just for logging:
					loss += l / lastindex(train_loader)
				end
        elseif !isnothing(sample_weights_loader) && !isnothing(train_loader)
            for ((x, y), sample_weights) in zip(train_loader, sample_weights_loader)
                local l = 0.0
                l, grad = Flux.withgradient(m -> logitcrossentropyweighted(m(x), y, sample_weights), nn)
                Flux.update!(opt_state, nn, grad[1])
                loss += l / lastindex(train_loader)
            end
        elseif !isnothing(data) && nn_arch != "Evidential Classification"
            x, y = data
            loss, grad = Flux.withgradient(m -> loss_function(m(x), y), nn)
            Flux.update!(opt_state, nn, grad[1])
		elseif !isnothing(data) && nn_arch == "Evidential Classification"
            x, y = data
            loss, grad = Flux.withgradient(m -> dirloss(y, m(x), e), nn)
            Flux.update!(opt_state, nn, grad[1])
        end

        trnlosses[e] = loss

        if !isfinite(loss)
            @warn "loss is $loss" e
            continue
        end

        # if mod(e, 2) == 1
        # 	# Report on train and test, only every 2nd epoch_idx:
        # 	@info "After Epoch $e" loss
        # @warn("After Epoch $e -> $(opt_state.weight.rule)\n $(opt_state.bias.rule)!")
        # end

        if abs(loss) < abs(least_loss)
            # @info("After Epoch $e -> New minimum loss $loss. Saving model weights.\n $(opt_state.weight.rule)")
            optim_params, re = Flux.destructure(nn)
            least_loss = loss
            last_improvement = e
        end

        if e - last_improvement >= 100
            @warn("After Epoch $e -> We're calling this converged.")
            break
        end

    end
    scatter(1:n_epochs, trnlosses, width=80, height=30)
    savefig("./$(nn_arch)_loss.pdf")

    # optim_params, re = Flux.destructure(nn)
    return optim_params, re
end