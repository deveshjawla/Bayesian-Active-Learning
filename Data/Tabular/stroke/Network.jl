l1, l2 = 12, 12 
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:48], 12, 4)
	b10 = reshape(nn_params[49:60], 12)

	w20 = reshape(nn_params[61:204], 12, 12)
	b20 = reshape(nn_params[205:216], 12)

	w31 = reshape(nn_params[217:240], 2, 12)

	model = Chain(Dense(w10, b10,relu), Dense(w20, b20,relu), Dense(w31, false))
	return model
end
num_params = 240