l1, l2, l3, l4 = 11, 11, 11, 11
nl1 = 11 * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output + n_output

num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:121], 11, 11)
	b0 = θ[122:132]
	W1 = reshape(θ[133:253], 11, 11)
	b1 = θ[254:264]
	W2 = reshape(θ[265:385], 11, 11)
	b2 = θ[386:396]
	W3 = reshape(θ[397:517], 11, 11)
	b3 = θ[518:528]
	W4 = reshape(θ[529:550], 2, 11)
	b4 = θ[551:552]
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