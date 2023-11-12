l1, l2 = 12, 12
nl1 = 22 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:66], 3, 22)
	b10 = reshape(nn_params[67:69], 3)
	w11 = reshape(nn_params[70:135], 3, 22)
	b11 = reshape(nn_params[136:138], 3)
	w12 = reshape(nn_params[139:204], 3, 22)
	b12 = reshape(nn_params[205:207], 3)
	w13 = reshape(nn_params[208:273], 3, 22)
	b13 = reshape(nn_params[274:276], 3)

	w20 = reshape(nn_params[277:312], 3, 12)
	b20 = reshape(nn_params[313:315], 3)
	w21 = reshape(nn_params[316:351], 3, 12)
	b21 = reshape(nn_params[352:354], 3)
	w22 = reshape(nn_params[355:390], 3, 12)
	b22 = reshape(nn_params[391:393], 3)
	w23 = reshape(nn_params[394:429], 3, 12)
	b23 = reshape(nn_params[430:432], 3)

	w31 = reshape(nn_params[433:456], 2, 12)
	b31 = reshape(nn_params[457:458], 2)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,identity), Dense(w11, b11, sin), Dense(w12, b12, tanh), Dense(w13, b13, relu)), 
		Parallel(vcat, Dense(w20, b20,identity), Dense(w21, b21, sin), Dense(w22, b22, tanh), Dense(w23, b23, relu)), 
		Dense(w31, b31), softmax)
	return model
end

num_params = 458