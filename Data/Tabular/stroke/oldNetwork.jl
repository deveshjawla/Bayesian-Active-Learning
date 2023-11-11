l1, l2 = 10, 10
nl1 = 10 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:100], 10, 10)
	b0 = θ[101:110]
	W1 = reshape(θ[111:210], 10, 10)
	b1 = θ[211:220]
	W2 = reshape(θ[221:240], 2, 10)
	b2 = θ[241:242]

	model = Chain(
		Dense(W0, b0, tanh),
		Dense(W1, b1, tanh),
		Dense(W2, b2,),
		softmax
	)
	return model
end

num_params = 242