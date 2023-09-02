l1, l2, = 22, 22
nl1 = 22 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:484], 22, 22)
	b0 = θ[485:506]
	W1 = reshape(θ[507:990], 22, 22)
	b1 = θ[991:1012]
	W2 = reshape(θ[1013:1056], 2, 22)
	b2 = θ[1057:1058]


	return Chain(
		Dense(W0, b0, mish),
		Dense(W1, b1, mish),
		Dense(W2, b2),
		softmax
	)
end

num_params = 1058

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