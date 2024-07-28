l1, l2 = 24, 24 
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:24], 6, 4)
	b10 = reshape(nn_params[25:30], 6)
	w11 = reshape(nn_params[31:54], 6, 4)
	b11 = reshape(nn_params[55:60], 6)
	w12 = reshape(nn_params[61:84], 6, 4)
	b12 = reshape(nn_params[85:90], 6)
	w13 = reshape(nn_params[91:114], 6, 4)
	b13 = reshape(nn_params[115:120], 6)

	w20 = reshape(nn_params[121:264], 6, 24)
	b20 = reshape(nn_params[265:270], 6)
	w21 = reshape(nn_params[271:414], 6, 24)
	b21 = reshape(nn_params[415:420], 6)
	w22 = reshape(nn_params[421:564], 6, 24)
	b22 = reshape(nn_params[565:570], 6)
	w23 = reshape(nn_params[571:714], 6, 24)
	b23 = reshape(nn_params[715:720], 6)

	w31 = reshape(nn_params[721:768], 2, 24)
	b31 = reshape(nn_params[769:770], 2)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,relu), Dense(w11, b11, relu), Dense(w12, b12, relu), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,relu), Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23, relu)), 
		Dense(w31, false))
	return model
end

num_params = 770