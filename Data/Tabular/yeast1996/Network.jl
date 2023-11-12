l1, l2 = 12, 12 
nl1 = 8 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:24], 3, 8)
	b10 = reshape(nn_params[25:27], 3)
	w11 = reshape(nn_params[28:51], 3, 8)
	b11 = reshape(nn_params[52:54], 3)
	w12 = reshape(nn_params[55:78], 3, 8)
	b12 = reshape(nn_params[79:81], 3)
	w13 = reshape(nn_params[82:105], 3, 8)
	b13 = reshape(nn_params[106:108], 3)

	w20 = reshape(nn_params[109:144], 3, 12)
	b20 = reshape(nn_params[145:147], 3)
	w21 = reshape(nn_params[148:183], 3, 12)
	b21 = reshape(nn_params[184:186], 3)
	w22 = reshape(nn_params[187:222], 3, 12)
	b22 = reshape(nn_params[223:225], 3)
	w23 = reshape(nn_params[226:261], 3, 12)
	b23 = reshape(nn_params[262:264], 3)

	w31 = reshape(nn_params[265:384], 10, 12)
	b31 = reshape(nn_params[385:394], 10)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,identity), Dense(w11, b11, sin), Dense(w12, b12, tanh), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,identity), Dense(w21, b21, sin), Dense(w22, b22, tanh), Dense(w23, b23, relu)), 
		Dense(w31, b31), softmax)
	return model
end

num_params = 394