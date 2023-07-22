l1, l2 = 4, 4
nl1 = 3 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:12], 4, 3)
	b0 = θ[13:16]
	W1 = reshape(θ[17:32], 4, 4)
	b1 = θ[33:36]
	W2 = reshape(θ[37:44], 2, 4)
	b2 = θ[45:46]

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2),
		softmax
	)
	return model
end

num_params = 46

