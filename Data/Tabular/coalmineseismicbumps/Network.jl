l1, l2, = 12, 12
nl1 = 11 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:132], 12, 11)
	b10 = reshape(nn_params[133:144], 12)

	w20 = reshape(nn_params[145:288], 12, 12)
	b20 = reshape(nn_params[289:300], 12)

	w31 = reshape(nn_params[301:324], 2, 12)

	model = Chain(Dense(w10, b10,relu), Dense(w20, b20,relu), Dense(w31, false))
	return model
end

num_params = 324