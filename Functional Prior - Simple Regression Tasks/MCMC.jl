using Flux
using StatsPlots
using Distributions
using Turing
using DelimitedFiles
PATH = @__DIR__
cd(PATH)
using ReverseDiff

activation_function = "mix"
temperatures = [1.0, 0.1, 0.001]
function_names = ["Cosine", "Polynomial", "Exponential", "Logarithmic", "sin(pisin)"]#  "Cosine", "Polynomial", "Exponential", "Logarithmic",


include("./MixActivationFunctionsNN2Layers.jl")

include("./MakeData.jl")
include("./../BayesianModel.jl")
for function_name in function_names
    for temperature in temperatures
        experiment = "custom_loglikelihood_MCMC_$(temperature)"

        xs, ys, Xline = make_data(function_name)

        num_params = lastindex(prior_std)

        N = 100
        ch1_timed = @timed sample(regressionBNN(hcat(xs...), ys, num_params, prior_std), NUTS(; adtype=AutoReverseDiff()), N)
        ch1 = ch1_timed.value
        elapsed = Float32(ch1_timed.time)
        # ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);

        weights = Array(MCMCChains.group(ch1, :nn_params).value) #get posterior MCMC samples for network weights
        println(size(weights))
        sigmas = Array(MCMCChains.group(ch1, :sigma).value)

        mkpath("./$(experiment)/Plots")
        writedlm("$(experiment)/$(function_name)_$(activation_function)_weights.csv", weights[:, :, 1], ',')
        writedlm("$(experiment)/$(function_name)_$(activation_function)_sigmas.csv", sigmas, ',')
        writedlm("$(experiment)/$(function_name)_$(activation_function)_elapsed.csv", elapsed, ',')

        # weights= readdlm("$(experiment)/$(function_name)_$(activation_function)_weights.csv", ',')
        # sigmas=readdlm("$(experiment)/$(function_name)_$(activation_function)_sigmas.csv", ',')

        pipeline_name = "$(function_name)_$(activation_function)"
        make_mcmc_regression_plots(experiment, pipeline_name, weights, sigmas, xs, ys, Xline)
    end
end
