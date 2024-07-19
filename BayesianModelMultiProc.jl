# function feedforward(x, θ::AbstractVector, network_shape::AbstractVector)
#     index = 1
#     weights = []
#     biases = []
#     for layer in network_shape
#         rows, cols, _ = layer
#         size = rows * cols
#         last_index_w = size + index - 1
#         last_index_b = last_index_w + rows
#         push!(weights, reshape(θ[index:last_index_w], rows, cols))
#         push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
#         index = last_index_b + 1
#     end
#     layers = []
#     for i in eachindex(network_shape)
#         push!(layers, Dense(weights[i],
#             biases[i],
#             eval(network_shape[i][3])))
#     end
#     nn = deepcopy(Chain(layers..., softmax))
# 	return nn(x)
# end

# Define a model on all processes
# @model function bayesnnMVG(x, y, location_prior, scale_prior)
# 	θ ~ MvNormal(location_prior, scale_prior)
# 	nn = feedforward(θ)(x)
# 	for i = 1:lastindex(y)
# 		y[i] ~ Categorical(nn[:, i])
# 	end
# end

# # General Turing specification for a BNN model.
# @model function bayesnnMVG(x, y, network_shape, num_params)
#     θ ~ MvNormal(zeros(num_params), ones(num_params))
#     preds = feedforward(x, θ, network_shape)
#     for i = 1:lastindex(y)
# 		y[i] ~ Categorical(preds[:, i])
# 	end
# end

# using InteractiveUtils
# @model function bayesnnMVG(x, y, num_params, warn=true)
# 	θ ~ MvNormal(zeros(num_params), ones(num_params))
# 	# @code_warntype feedforward(θ)
# 	nn = feedforward(θ)
# 	preds = nn(x)
# 	for i = 1:lastindex(y)
# 		y[i] ~ Categorical(preds[:, i])
# 	end
# end

@model function temperedBNN(x, y, location, scale, Temp)
    θ ~ MvNormal(location, scale)
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
@model function classweightedBNN(x, y, location, scale, weights_vector)
    θ ~ MvNormal(location, scale)
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

@model function BNN(x, y, location, scale)
    # # Hyper priors
    # n_weights_input = num_params - lastindex(init_params)
    # input_hyperprior ~ filldist(Exponential(0.2), n_weights_input)
    # θ_input ~ MvNormal(zeros(n_weights_input), input_hyperprior)
    # θ_hidden ~ MvNormal(0 .* init_params, init_params)

    θ ~ MvNormal(location, scale)
    # @code_warntype feedforward(θ)
    nn = feedforward(θ)
    # nn = feedforward(θ_input, θ_hidden)
    preds = nn(x)

    if n_output == 1
        sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
        for i = 1:lastindex(y)
            y[i] ~ Normal(preds[i], sigma)
        end
    else

        for i = 1:lastindex(y)
            y[i] ~ Categorical(preds[:, i])
        end
    end
end

# @model function bayesnnMVG(x, y, num_params, warn = true)
#     θ ~ MvNormal(zeros(num_params), ones(num_params))
#     nn = feedforward(x, θ)
#     for i = 1:lastindex(y)
#         y[i] ~ Categorical(nn[:, i])
#     end
# end

# @model function bayesnnMVG(x, y, num_params)
#     θ ~ MvNormal(zeros(num_params), ones(num_params))
#     nn = feedforward(θ)(x)
# 	preds = deepcopy(collect.(eachcol(nn)))
#     # labels ~ arraydist(LazyArray(@~ Categorical.(preds)))
#     labels ~ Product(Categorical.(preds))
# end


# @model function bayesnnMVG(x, y, init_params, num_params)
#     θ ~ MvNormal(zeros(num_params), init_params)
#     nn = feedforward(θ)(x)
# 	preds = deepcopy(collect.(eachcol(nn)))
# 	labels = deepcopy(vec(permutedims(y)))
#     # labels ~ arraydist(LazyArray(@~ Categorical.(preds)))
#     labels ~ Product(Categorical.(preds))
# end



