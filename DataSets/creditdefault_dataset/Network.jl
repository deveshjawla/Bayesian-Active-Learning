l1, l2 = 23, 23
nl1 = 23 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:529], 23, 23)
	b0 = θ[530:552]
	W1 = reshape(θ[553:1081], 23, 23)
	b1 = θ[1082:1104]
	W2 = reshape(θ[1105:1150], 2, 23)
	b2 = θ[1151:1152]

	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2),
		softmax
	)
	return model
end

num_params = 1152