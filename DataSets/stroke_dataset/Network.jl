###
### Dense Network specifications(Functional Model)
###

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:200], 20, 10)
	b0 = θ[201:220]
	W1 = reshape(θ[221:620], 20, 20)
	b1 = θ[621:640]
	W2 = reshape(θ[641:1040], 20, 20)
	b2 = θ[1041:1060]
	W3 = reshape(θ[1061:1460], 20, 20)
	b3 = θ[1461:1480]
	W4 = reshape(θ[1481:1520], 2, 20)
	b4 = θ[1521:1522]
	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2, relu),
		Dense(W3, b3, relu),
		Dense(W4, b4),
		softmax
	)
	return model
end

# nn_initial = Chain(Dense(input_size, l1, relu), Dense(l1, l2, relu), Dense(l2, l3, relu), Dense(l3, l4, relu), Dense(l4, n_output, relu), softmax)

# # Extract weights and a helper function to reconstruct NN from weights
# parameters_initial, reconstruct = Flux.destructure(nn_initial)

# total_num_params = length(parameters_initial) # number of paraemters in NN