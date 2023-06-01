l1, l2 = 4, 4
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:16], 4, 4)
	b0 = θ[17:20]
	W1 = reshape(θ[21:36], 4, 4)
	b1 = θ[37:40]
	W2 = reshape(θ[41:52], 3, 4)
	b2 = θ[53:55]

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2, relu),
		softmax
	)
	return model
end

num_params = 55