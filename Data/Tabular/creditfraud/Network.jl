l1, l2 = 12, 12
nl1 = 28 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:336], 12, 28)
	b10 = reshape(nn_params[337:348], 12)

	w20 = reshape(nn_params[349:492], 12, 12)
	b20 = reshape(nn_params[493:504], 12)

	w31 = reshape(nn_params[505:528], 2, 12)

	model = Chain(Dense(w10, b10,relu), Dense(w20, b20,relu), Dense(w31, false))
	return model
end

num_params = 528