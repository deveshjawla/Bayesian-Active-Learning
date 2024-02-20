l1, l2 = 24, 24 
nl1 = 8 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:48], 6, 8)
	b10 = reshape(nn_params[49:54], 6)
	w11 = reshape(nn_params[55:102], 6, 8)
	b11 = reshape(nn_params[103:108], 6)
	w12 = reshape(nn_params[109:156], 6, 8)
	b12 = reshape(nn_params[157:162], 6)
	w13 = reshape(nn_params[163:210], 6, 8)
	b13 = reshape(nn_params[211:216], 6)

	w20 = reshape(nn_params[217:360], 6, 24)
	b20 = reshape(nn_params[361:366], 6)
	w21 = reshape(nn_params[367:510], 6, 24)
	b21 = reshape(nn_params[511:516], 6)
	w22 = reshape(nn_params[517:660], 6, 24)
	b22 = reshape(nn_params[661:666], 6)
	w23 = reshape(nn_params[667:810], 6, 24)
	b23 = reshape(nn_params[811:816], 6)

	w31 = reshape(nn_params[817:1056], 10, 24)
	b31 = reshape(nn_params[1057:1066], 10)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,relu), Dense(w11, b11, relu), Dense(w12, b12, relu), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,relu), Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23, relu)), 
		Dense(w31, b31), softmax)
	return model
end

num_params = 1066