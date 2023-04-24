# # Define a model on all processes
# @everywhere @model function bayesnnMVG(x, y, μ_prior, σ_prior)
# 	θ ~ MvNormal(μ_prior, σ_prior)
# 	nn = feedforward(θ)

# 	nn_output = nn(x)
# 	for i = 1:lastindex(y)
# 		y[i] ~ Categorical(nn_output[:, i])
# 	end
# end

# @everywhere @model function bayesnnMVG(x, y, μ_prior, σ_prior, reconstruct)
#     θ ~ MvNormal(μ_prior, σ_prior)
#     nn = reconstruct(θ)
#     ŷ = nn(x)
#     for i = 1:lastindex(y)
#         y[i] ~ Categorical(ŷ[:, i])
#     end
# end

# @everywhere @model function bayesnnMVG(x, y, μ_prior, σ_prior, reconstruct)
#     θ ~ MvNormal(μ_prior, σ_prior)
#     nn = reconstruct(θ)
#     ŷ = nn(x)
# 	y ~ arraydist(LazyArray(@~ Categorical.(ŷ)))
# end

@model function bayesnnMVG(x, y, μ_prior, σ_prior)
	θ ~ MvNormal(μ_prior, σ_prior)
	nn = feedforward(θ)

	nn_output = nn(x)
	for i = 1:lastindex(y)
		y[i] ~ Categorical(nn_output[:, i])
	end
end

@model function bayesnnMVG(x, y, μ_prior, σ_prior, reconstruct)
    θ ~ MvNormal(μ_prior, σ_prior)
    nn = reconstruct(θ)
    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end



