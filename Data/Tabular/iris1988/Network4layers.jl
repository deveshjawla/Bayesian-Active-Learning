l1, l2, l3, l4 = 4, 4, 4, 4
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output + n_output

num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:16], 4, 4)
	b0 = θ[17:20]
	W1 = reshape(θ[21:36], 4, 4)
	b1 = θ[37:40]
	W2 = reshape(θ[41:56], 4, 4)
	b2 = θ[57:60]
	W3 = reshape(θ[61:76], 4, 4)
	b3 = θ[77:80]
	W4 = reshape(θ[81:92], 3, 4)
	b4 = θ[93:95]
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