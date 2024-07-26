function feedforward(nn_params::AbstractVector)
    w11 = reshape(nn_params[1:3], 3, 1)
    b11 = reshape(nn_params[4:6], 3)
    w12 = reshape(nn_params[7:7], 1, 1)
    b12 = reshape(nn_params[8:8], 1)
    w13 = reshape(nn_params[9:9], 1, 1)
    b13 = reshape(nn_params[10:10], 1)

    w21 = reshape(nn_params[11:15], 1, 5)
    b21 = reshape(nn_params[16:16], 1)
    w22 = reshape(nn_params[17:21], 1, 5)
    b22 = reshape(nn_params[22:22], 1)
    w23 = reshape(nn_params[23:27], 1, 5)
    b23 = reshape(nn_params[28:28], 1)

    w31 = reshape(nn_params[29:31], 1, 3)
    b31 = reshape(nn_params[32:32], 1)
    nn = Chain(
        Parallel(vcat, Dense(w11, b11, relu), Dense(w12, b12, sin), Dense(w13, b13, identity)),
        Parallel(vcat, Dense(w21, b21, relu), Dense(w22, b22, sin), Dense(w23, b23, identity)),
        Dense(w31, b31))
    return nn
end

prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (1 + 5)) * ones(10), sqrt(2 / (5 + 3)) * ones(18), sqrt(2 / (3 + 1)) * ones(4)))
# prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + n_output)) * ones(n_output_layer))
# prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
