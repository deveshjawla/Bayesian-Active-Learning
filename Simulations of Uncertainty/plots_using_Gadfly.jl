using CSV, DataFrames
using Flux
using Distributions
using Turing
using ProgressMeter
using DelimitedFiles
PATH = @__DIR__
cd(PATH)
# n_input = 2
# n_output = 1

# l1 = 4
# nl1 = n_input * l1 + l1
# n_output_layer = l1 * n_output

# total_num_params = nl1 + n_output_layer

# function feedforward(θ::AbstractVector)
#     w1 = reshape(θ[1:8], 4, 2)
#     b1 = reshape(θ[9:12], 4)

#     w3 = reshape(θ[13:16], 1, 4)

#     model = Chain(Dense(w1, b1, relu), Dense(w3, false))
#     return model
# end


# prior_std = sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + n_output)) * ones(n_output_layer))
# num_params = lastindex(prior_std)

# @model function bayesian_nn(xs, ys, num_params, prior_variances)
#     θ ~ MvNormal(zeros(Float32, num_params), prior_variances) #Prior
#     nn = feedforward(θ)
#     preds = nn(xs)

#     sigma ~ Gamma(0.01, 1 / 0.01) # Prior for the variance
#     for i = 1:lastindex(ys)
#         ys[i] ~ Normal(preds[i], sigma)
#     end
# end

# N = 100

# df = CSV.read("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/correlations_dataframe.csv", DataFrame)
# df = filter(row -> !isnan(row.Correlation), df)

# groups = deepcopy(groupby(df, [:Case, :Pair]))

# @showprogress for g in groups
#     xs = permutedims(Matrix(g[!, [:NumCategories, :TotalSamples]]))
#     ys = permutedims(Matrix(g[!, [:Correlation]]))

#     ch_timed = @timed sample(bayesian_nn(xs, ys, num_params, prior_std), NUTS(), N)
#     ch = ch_timed.value
#     elapsed = ch_timed.time

#     plot(Array(MCMCChains.group(ch, :lp).value)[:, 1, 1])

#     weights = Array(MCMCChains.group(ch, :θ).value) #get posterior MCMC samples for network weights
#     println(size(weights))
#     sigmas = Array(MCMCChains.group(ch, :sigma).value)

#     # 
#     # writedlm("BNN regression/$(case)_$(pair)_weights.csv", weights[:, :, 1], ',')
#     # writedlm("BNN regression/$(case)_$(pair)_sigmas.csv", sigmas, ',')
#     # writedlm("BNN regression/$(case)_$(pair)_elapsed.csv", elapsed, ',')

#     # weights= readdlm("BNN regression/$(case)_$(pair)_weights.csv", ',')
#     # sigmas=readdlm("BNN regression/$(case)_$(pair)_sigmas.csv", ',')

#     test_x = [3 4 5 6; 100 100 100 100]

#     posterior_predictive_mean_samples = []
#     posterior_predictive_full_samples = []

#     for i in 1:N
#         W = weights[i, :, 1]
#         sigma = sigmas[i, 1, 1]
#         nn = feedforward(vec(W))
#         predictive_distribution = vec(Normal.(nn(test_x), sigma))
#         postpred_full_sample = rand(Product(predictive_distribution))
#         push!(posterior_predictive_mean_samples, mean.(predictive_distribution))
#         push!(posterior_predictive_full_samples, postpred_full_sample)
#     end

#     posterior_predictive_mean_samples = hcat(posterior_predictive_mean_samples...)

#     pp_mean = mean(posterior_predictive_mean_samples, dims=2)[:]
#     # pp_mean_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_mean_samples, dims=2)[:]
#     # pp_mean_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_mean_samples, dims=2)[:]

#     # posterior_predictive_full_samples = hcat(posterior_predictive_full_samples...)
#     # pp_full_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_full_samples, dims=2)[:]
#     # pp_full_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_full_samples, dims=2)[:]

#     # mean_pred, aleatoric_upper, aleatopric_lower = (pp_mean, pp_mean .- pp_full_lower, pp_full_upper .- pp_mean) #label="Full posterior predictive distribution"

#     # epistemic_upper, epistemic_lower = (pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean)# label="Posterior predictive mean distribution (epistemic uncertainty)"
# 	new_df = DataFrame(:Case=> collect(g.Case)[1:4], :NumCategories => vec(test_x[1, :]), :TotalSamples => vec(test_x[2, :]), :Pair => collect(g.Pair)[1:4], :Correlation => pp_mean)
# 	append!(df, new_df)
# end

# CSV.write("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/correlations_dataframe_with_extrapolations.csv", df)

df = CSV.read("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/correlations_dataframe.csv", DataFrame)
df = filter(row -> row.NumCategories .== 2, df)

using Gadfly, Cairo, Fontconfig
using Printf
# theme = Theme(key_position=:none)
# Gadfly.push_theme(theme)
experiment = "TotalSamples"
mkpath("./Plots/$(experiment)")
groups  = deepcopy(groupby(df, [:Case, :NumCategories]))
for g in groups
	plt = Gadfly.plot(g, x=:TotalSamples, y=:Correlation, color=:Pair, Gadfly.Geom.point(), Gadfly.Geom.line(), Scale.x_log10)
	plt |> PDF("./Plots/$(experiment)/Case=$(g.Case[1])_$(g.NumCategories[1]).pdf", 600pt, 600pt)
end