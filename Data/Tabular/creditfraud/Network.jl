l1, l2 = 28, 28
nl1 = 28 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:784], 28, 28)
	b0 = θ[785:812]
	W1 = reshape(θ[813:1596], 28, 28)
	b1 = θ[1597:1624]
	W2 = reshape(θ[1625:1680], 2, 28)
	b2 = θ[1681:1682]

	model = Chain(
		Dense(W0, b0, mish),
		Dense(W1, b1, mish),
		Dense(W2, b2),
		softmax
	)
	return model
end

num_params = 1682