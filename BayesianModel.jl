@model function temperedBNN(x, y, μ_Gaussian, Σ_Gaussian, Temp)
    θ ~ MvNormal(μ_Gaussian, Σ_Gaussian)
    # @code_warntype feedforward(θ)
    nn = feedforward(θ)
    # nn = feedforward(θ_input, θ_hidden)
    preds = nn(x)
    if n_input == 1
        sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
        for i = 1:lastindex(y)
            loglik = loglikelihood(Normal(preds[i], sigma), y[i]) / Temp
            Turing.@addlogprob!(loglik)
        end
    else

        for i = 1:lastindex(y)
            loglik = loglikelihood(Categorical(preds[:, i]), y[i]) / Temp
            Turing.@addlogprob!(loglik)
        end
    end
end

"""
weights is a vector of the sample weights according to their importance or proportion in the dataset
"""
@model function classweightedBNN(x, y, μ_Gaussian, Σ_Gaussian, weights_vector)
    θ ~ MvNormal(μ_Gaussian, Σ_Gaussian)
    # θ ~ Product([Laplace(0,1) for _ in 1:num_params])
    # @code_warntype feedforward(θ)
    nn = feedforward(θ)
    # nn = feedforward(θ_input, θ_hidden)
    preds = nn(x)
    for i = 1:lastindex(y)
        loglik = loglikelihood(Categorical(preds[:, i]), y[i]) * weights_vector[i]
        Turing.@addlogprob!(loglik)
    end
end

@model function BNN(x, y, μ_Gaussian, Σ_Gaussian)
    θ ~ MvNormal(μ_Gaussian, Σ_Gaussian)
    # @code_warntype feedforward(θ)
    nn = feedforward(θ)
    preds = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(preds[:, i])
    end
end

@model function regressionBNN(x, y, μ_Gaussian, Σ_Gaussian; α_Gamma=0.1, θ_Gamma=10)
    θ ~ MvNormal(μ_Gaussian, Σ_Gaussian)
    nn = feedforward(θ)
    preds = nn(x)

    sigma ~ Gamma(α_Gamma, θ_Gamma) # Prior for the variance
    for i = 1:lastindex(y)
        y[i] ~ Normal(preds[i], sigma)
    end
end

@model function softmax_bnn_noise_x(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    noise ~ MvNormal(zeros(2), ones(2))
    preds = nn(x .+ noise)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(softmax(preds[:, i]))
    end
end
@model function softmax_bnn_noise_y(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    preds = nn(x)
    noise_y ~ Gamma(0.1, 10)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(softmax(preds[:, i] .+ noise_y))
    end
end
@model function softmax_bnn_noise_xy(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    noise ~ MvNormal(zeros(2), ones(2))
    preds = nn(x .+ noise)
    noise_y ~ Product([Gamma(0.1, 10) for _ in 1:3])
    # noise_y ~ MvNormal(zeros(3), ones(3))
    for i = 1:lastindex(y)
        y[i] ~ Categorical(softmax(preds[:, i] .+ noise_y))
    end
end
@model function softmax_bnn(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    preds = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(softmax(preds[:, i]))
    end
end

@model function softplus_bnn_noise_x(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    noise ~ MvNormal(zeros(2), ones(2))
    preds = nn(x .+ noise)
    for i = 1:lastindex(y)
        y[i] ~ Categorical((preds[:, i]) ./ sum(preds[:, i]))
    end
end
@model function softplus_bnn_noise_y(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    preds = nn(x)
    noise_y ~ Gamma(0.1, 10)
    for i = 1:lastindex(y)
        y[i] ~ Categorical((preds[:, i] .+ noise_y) ./ sum(preds[:, i] .+ noise_y))
    end
end
@model function softplus_bnn_noise_xy(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    noise ~ MvNormal(zeros(2), ones(2))
    preds = nn(x .+ noise)
    noise_y ~ Product([Gamma(0.1, 10) for _ in 1:3])
    # noise_y ~ MvNormal(zeros(3), ones(3))
    for i = 1:lastindex(y)
        y[i] ~ Categorical((preds[:, i] .+ noise_y) ./ sum((preds[:, i] .+ noise_y)))
    end
end
@model function softplus_bnn(x, y, num_params, scale)
    θ ~ MvNormal(zeros(num_params), scale)
    nn = feedforward(θ)
    preds = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(preds[:, i] ./ sum(preds[:, i]))
    end
end