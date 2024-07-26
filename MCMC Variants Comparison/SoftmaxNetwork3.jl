function feedforward(θ::AbstractVector)
    W0 = reshape(θ[1:16], 8, 2)
    b0 = θ[17:24]
    W1 = reshape(θ[25:88], 8, 8)
    b1 = θ[89:96]
    W2 = reshape(θ[97:120], 3, 8)

    model = Chain(
        Dense(W0, b0, relu),
        Dense(W1, b1, relu),
        Dense(W2, false)
    )
    return model
end