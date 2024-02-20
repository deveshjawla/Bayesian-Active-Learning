l1, l2, l3, l4 = 8, 8, 8, 8
nl1 = 8 * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output + n_output

num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:64], 8, 8)
	b0 = θ[65:72]
	W1 = reshape(θ[73:136], 8, 8)
	b1 = θ[137:144]
	W2 = reshape(θ[145:208], 8, 8)
	b2 = θ[209:216]
	W3 = reshape(θ[217:280], 8, 8)
	b3 = θ[281:288]
	W4 = reshape(θ[289:368], 10, 8)
	b4 = θ[369:378]
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