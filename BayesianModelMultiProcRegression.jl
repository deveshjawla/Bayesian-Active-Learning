@model function temperedBNN(x, y, location, scale, Temp)
	θ ~ MvNormal(location, scale)
	# @code_warntype feedforward(θ)
	nn = feedforward(θ)
	# nn = feedforward(θ_input, θ_hidden)
	preds = nn(x)
	sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
	for i = 1:lastindex(y)
		loglik = loglikelihood(Normal(preds[i], sigma), y[i])/Temp
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
	sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
	for i = 1:lastindex(y)
		y[i] ~ Normal(preds[i], sigma)
	end
end