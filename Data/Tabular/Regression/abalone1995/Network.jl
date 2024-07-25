l1, l2 = 9, 9 
nl1 = 7 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w11 = reshape(nn_params[1:21], 3, 7)
	b11 = reshape(nn_params[22:24], 3)
	w12 = reshape(nn_params[25:45], 3, 7)
	b12 = reshape(nn_params[46:48], 3)
	w13 = reshape(nn_params[49:69], 3, 7)
	b13 = reshape(nn_params[70:72], 3)

	w21 = reshape(nn_params[73:99], 3, 9)
	b21 = reshape(nn_params[100:102], 3)
	w22 = reshape(nn_params[103:129], 3, 9)
	b22 = reshape(nn_params[130:132], 3)
	w23 = reshape(nn_params[133:159], 3, 9)
	b23 = reshape(nn_params[160:162], 3)

	w31 = reshape(nn_params[164:171], 1, 9)
	b31 = reshape(nn_params[172:172], 1)

	model = Chain(
		Parallel(vcat, Dense(w11, b11,relu), Dense(w12, b12, relu), Dense(w13, b13, relu)), Parallel(vcat, Dense(w21, b21, relu), Dense(w22, b22, relu), Dense(w23, b23,relu)), Dense(w31, b31))
	return model
end

num_params = 172