l1, l2 = 12, 12
nl1 = 28 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:84], 3, 28)
	b10 = reshape(nn_params[85:87], 3)
	w11 = reshape(nn_params[88:171], 3, 28)
	b11 = reshape(nn_params[172:174], 3)
	w12 = reshape(nn_params[175:258], 3, 28)
	b12 = reshape(nn_params[259:261], 3)
	w13 = reshape(nn_params[262:345], 3, 28)
	b13 = reshape(nn_params[346:348], 3)

	w20 = reshape(nn_params[349:384], 3, 12)
	b20 = reshape(nn_params[385:387], 3)
	w21 = reshape(nn_params[388:423], 3, 12)
	b21 = reshape(nn_params[424:426], 3)
	w22 = reshape(nn_params[427:462], 3, 12)
	b22 = reshape(nn_params[463:465], 3)
	w23 = reshape(nn_params[466:501], 3, 12)
	b23 = reshape(nn_params[502:504], 3)

	w31 = reshape(nn_params[505:528], 2, 12)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,relu), Dense(w11, b11, relu), Dense(w12, b12, relu), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,relu), Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23, relu)), 
		Dense(w31, false))
	return model
end

num_params = 528