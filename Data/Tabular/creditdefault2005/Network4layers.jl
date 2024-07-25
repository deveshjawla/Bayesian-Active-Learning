l1, l2, l3, l4 = 22, 8, 8, 8
nl1 = 22 * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output

num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:484], 22, 22)
	b0 = θ[485:506]
	W1 = reshape(θ[507:682], 8, 22)
	b1 = θ[683:690]
	W2 = reshape(θ[691:754], 8, 8)
	b2 = θ[755:762]
	W3 = reshape(θ[763:826], 8, 8)
	b3 = θ[827:834]
	W4 = reshape(θ[835:850], 2, 8)
	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2, relu),
		Dense(W3, b3, relu),
		Dense(W4, false)
	)
	return model
end