l1, l2, = 12, 12
nl1 = 11 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(nn_params::AbstractVector)
	w10	= reshape(nn_params[1:33], 3, 11)
	b10 = reshape(nn_params[34:36], 3)
	w11 = reshape(nn_params[37:69], 3, 11)
	b11 = reshape(nn_params[70:72], 3)
	w12 = reshape(nn_params[73:105], 3, 11)
	b12 = reshape(nn_params[106:108], 3)
	w13 = reshape(nn_params[109:141], 3, 11)
	b13 = reshape(nn_params[142:144], 3)

	w20 = reshape(nn_params[145:180], 3, 12)
	b20 = reshape(nn_params[181:183], 3)
	w21 = reshape(nn_params[184:219], 3, 12)
	b21 = reshape(nn_params[220:222], 3)
	w22 = reshape(nn_params[223:258], 3, 12)
	b22 = reshape(nn_params[259:261], 3)
	w23 = reshape(nn_params[262:297], 3, 12)
	b23 = reshape(nn_params[298:300], 3)

	w31 = reshape(nn_params[301:324], 2, 12)
	b31 = reshape(nn_params[325:326], 2)

	model = Chain(
		Parallel(vcat, Dense(w10, b10,identity), Dense(w11, b11,mish), Dense(w12, b12, tanh), Dense(w13, b13, sin)), Parallel(vcat, Dense(w20, b20,identity), Dense(w21, b21, mish), Dense(w22, b22, tanh), Dense(w23, b23,sin)), Dense(w31, b31), softmax)
	return model
end

num_params = 326

# l1, l2, l3, l4 = 22, 22, 22, 22
# nl1 = 22 * l1 + l1
# nl2 = l1 * l2 + l2
# nl3 = l2 * l3 + l3
# nl4 = l3 * l4 + l4
# n_output_layer = l4 * n_output + n_output

# total_num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

# function feedforward(θ::AbstractVector)
# 	W0 = reshape(θ[1:484], 22, 22)
# 	b0 = θ[485:506]
# 	W1 = reshape(θ[507:990], 22, 22)
# 	b1 = θ[991:1012]
# 	W2 = reshape(θ[1013:1496], 22, 22)
# 	b2 = θ[1497:1518]
# 	W3 = reshape(θ[1519:2002], 22, 22)
# 	b3 = θ[2003:2024]
# 	W4 = reshape(θ[2025:2068], 2, 22)
# 	b4 = θ[2069:2070]

# 	return Chain(
# 		Dense(W0, b0, mish),
# 		Dense(W1, b1, mish),
# 		Dense(W2, b2, mish),
# 		Dense(W3, b3, mish),
# 		Dense(W4, b4),
# 		softmax
# 	)
# end

# num_params = 2070