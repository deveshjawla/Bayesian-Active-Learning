# Define a model on all processes
@everywhere @model function bayesnnMVG(x, y, location_prior, scale_prior)
	θ ~ MvNormal(location_prior, scale_prior)
	nn = feedforward(θ)

	nn_output = nn(x)
	for i = 1:lastindex(y)
		y[i] ~ Categorical(nn_output[:, i])
	end
end

@everywhere @model function bayesnnMVG(x, y, location_prior, scale_prior, reconstruct)
    θ ~ MvNormal(location_prior, scale_prior)
    nn = reconstruct(θ)
    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end

# @everywhere @model function bayesnnMVG(x, y, location_prior, scale_prior, reconstruct)
#     θ ~ MvNormal(location_prior, scale_prior)
#     nn = reconstruct(θ)
#     ŷ = nn(x)
# 	y ~ arraydist(LazyArray(@~ Categorical.(ŷ)))
# end




