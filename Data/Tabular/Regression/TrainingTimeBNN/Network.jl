l1, l2 = 5, 5
nl1 = 2 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:10], 5, 2)
	b0 = θ[11:15]
	W1 = reshape(θ[16:40], 5, 5)
	b1 = θ[41:45]
	W2 = reshape(θ[46:50], 1, 5)

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, false)
	)
	return model
end

num_params = 50