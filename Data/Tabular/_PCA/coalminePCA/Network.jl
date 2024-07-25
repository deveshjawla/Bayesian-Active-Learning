l1, l2 = 4, 4
nl1 = 2 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:8], 4, 2)
	b0 = θ[9:12]
	W1 = reshape(θ[13:28], 4, 4)
	b1 = θ[29:32]
	W2 = reshape(θ[33:40], 2, 4)

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, false)
	)
	return model
end

num_params = 40

