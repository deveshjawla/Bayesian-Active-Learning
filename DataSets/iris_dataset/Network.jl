###
### Dense Network specifications(Functional Model)
###


function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:20], 5, 4)
	b0 = θ[21:25]
	W1 = reshape(θ[26:50], 5, 5)
	b1 = θ[51:55]
	W2 = reshape(θ[56:80], 5, 5)
	b2 = θ[81:85]
	W3 = reshape(θ[86:110], 5, 5)
	b3 = θ[111:115]
	W4 = reshape(θ[116:130], 3, 5)
	b4 = θ[131:133]
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
