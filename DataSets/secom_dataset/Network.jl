l1, l2 = 30, 30
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:13620], 30, 454)
	b0 = θ[13621:13650]
	W1 = reshape(θ[13651:14550], 30, 30)
	b1 = θ[14551:14580]
	W2 = reshape(θ[14581:14640], 2, 30)
	b2 = θ[14641:14642]

	model = Chain(
		Dense(W0, b0, mish),
		Dense(W1, b1, mish),
		Dense(W2, b2),
		softmax
	)
	return model
end

function feedforward(θ_input::AbstractVector, θ_hidden::AbstractVector)
	W0 = reshape(θ_input, 30, 454)
	b0 = θ_hidden[1:30]
	W1 = reshape(θ_hidden[31:930], 30, 30)
	b1 = θ_hidden[931:960]
	W2 = reshape(θ_hidden[961:1020], 2, 30)
	b2 = θ_hidden[1021:1022]

	model = Chain(
		Dense(W0, b0, mish),
		Dense(W1, b1, mish),
		Dense(W2, b2),
		softmax
	)
	return model
end

num_params = 14642