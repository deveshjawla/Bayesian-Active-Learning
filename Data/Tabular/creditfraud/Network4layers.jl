l1, l2, l3, l4 = 28, 8, 8, 8
nl1 = 28 * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output + n_output

num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:784], 28, 28)
	b0 = θ[785:812]
	W1 = reshape(θ[813:1036], 8, 28)
	b1 = θ[1037:1044]
	W2 = reshape(θ[1045:1108], 8, 8)
	b2 = θ[1109:1116]
	W3 = reshape(θ[1117:1180], 8, 8)
	b3 = θ[1181:1188]
	W4 = reshape(θ[1189:1204], 2, 8)
	b4 = θ[1205:1206]
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