l1, l2 = 17, 17
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:289], 17, 17)
	b0 = θ[290:306]
	W1 = reshape(θ[307:595], 17, 17)
	b1 = θ[596:612]
	W2 = reshape(θ[613:646], 2, 17)

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, false)
	)
	return model
end

num_params = 648