"""
Compute the optimum weights of the RBF NN using Stochastic Gradient Descent method AdaBelief. Uses CosAnneal for scheduling the learning rate η and automatically saves the best model obtained during the epochs.

Parameters
X       - The training set inputs.
y       - The training set labels.

Keyword Arguements
lambda  - The L2 regularization parameter
n_epochs - Number of Epochs

Returns
optim_theta - The learned weights of the RBF-NN
RBFNN - The RBF-NN architecture
"""
function computeWeights(X::Array{Float32,2}, y::Array{Int64,1}, numRBFNeurons, numCats, verbose; lambda=0.0, n_epochs=1000)
    if (verbose)
        println("3. Learn output weights.\n")
    end
    nn = Dense(numRBFNeurons, numCats; bias=false)

    y = Flux.onehotbatch(vec(y), 1:numCats)

    opt = OptimiserChain(WeightDecay(lambda), AdaBelief())
	s = ParameterSchedulers.Stateful(CosAnneal(0.1, 1e-6, 100))
    opt_state = Flux.setup(opt, nn)

    # Train it
    least_loss = Inf32
    last_improvement = 0
    optim_theta = 0
    re = 0

    trnlosses = []
    for e in 1:n_epochs
        local loss = 0.0f0

		# global opt_state, nn
        Flux.adjust!(opt_state, ParameterSchedulers.next!(s))
        loss, grad = Flux.withgradient(m -> Flux.Losses.logitcrossentropy(m(X), y), nn)
        push!(trnlosses, loss)
        Flux.update!(opt_state, nn, grad[1])
		
        # if mod(e, 2) == 1
        # 	# Report on train and test, only every 2nd epoch_idx:
        # 	@info "After Epoch $e" loss
		# @warn("After Epoch $e -> $(opt_state.weight.rule)\n $(opt_state.bias.rule)!")
        # end

        if abs(loss) < abs(least_loss)
            # @info("After Epoch $e -> New minimum loss $loss. Saving model weights.\n $(opt_state.weight.rule)")
            optim_theta, re = Flux.destructure(nn)
            least_loss = loss
            last_improvement = e
        end

        if e - last_improvement >= 100
            @warn("After Epoch $e -> We're calling this converged.")
            break
        end

    end
    scatter(trnlosses, width=80, height=30)
    savefig("./loss_cruve.pdf")

    return optim_theta, re
end