l1, l2 = 12, 12 
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:12], 3, 4)
	b10 = reshape(nn_params[13:15], 3)
	w11 = reshape(nn_params[16:27], 3, 4)
	b11 = reshape(nn_params[28:30], 3)
	w12 = reshape(nn_params[31:42], 3, 4)
	b12 = reshape(nn_params[43:45], 3)
	w13 = reshape(nn_params[46:57], 3, 4)
	b13 = reshape(nn_params[58:60], 3)

	w20 = reshape(nn_params[61:96], 3, 12)
	b20 = reshape(nn_params[97:99], 3)
	w21 = reshape(nn_params[100:135], 3, 12)
	b21 = reshape(nn_params[136:138], 3)
	w22 = reshape(nn_params[139:174], 3, 12)
	b22 = reshape(nn_params[175:177], 3)
	w23 = reshape(nn_params[178:213], 3, 12)
	b23 = reshape(nn_params[214:216], 3)

	w31 = reshape(nn_params[217:240], 2, 12)
	b31 = reshape(nn_params[241:242], 2)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,relu), Dense(w11, b11, relu), Dense(w12, b12, relu), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,relu), Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23, relu)), 
		Dense(w31, b31), softmax)
	return model
end

num_params = 242