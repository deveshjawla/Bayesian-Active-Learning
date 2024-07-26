function make_mcmc_regression_plots(experiment, pipeline_name, weights, sigmas, xs, ys, Xline)

    posterior_predictive_mean_samples = []
    posterior_predictive_full_samples = []

	N = size(weights, 1)
    for i in 1:N
        W = weights[i, :, 1]
        sigma = sigmas[i, :, 1]
        nn = nn_forward(vec(W))
        predictive_distribution = vec(Normal.(nn(permutedims(Xline)), sigma[1]))
        postpred_full_sample = rand(Product(predictive_distribution))
        push!(posterior_predictive_mean_samples, mean.(predictive_distribution))
        push!(posterior_predictive_full_samples, postpred_full_sample)
    end

    posterior_predictive_mean_samples = hcat(posterior_predictive_mean_samples...)

    pp_mean = mean(posterior_predictive_mean_samples, dims=2)[:]
    pp_mean_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_mean_samples, dims=2)[:]
    pp_mean_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_mean_samples, dims=2)[:]

    posterior_predictive_full_samples = hcat(posterior_predictive_full_samples...)
    pp_full_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_full_samples, dims=2)[:]
    pp_full_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_full_samples, dims=2)[:]

    plt = plot(Xline[:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), legend=:none, label="Full posterior predictive distribution", fmt=:pdf, size=(600, 400), dpi=600)

    plot!(Xline[:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (epistemic uncertainty)")


    lp, maxInd = findmax(ch1[:lp])

    params, internals = ch1.name_map
    bestParams = map(x -> ch1[x].data[maxInd], params[1:num_params])
    plot!(Xline, vec(nn_forward(bestParams)(permutedims(Xline))), seriestype=:line, label="MAP Estimate", color=:navy)

    scatter!(xs, ys, color=:green, markersize=2.0, label="Training data", markerstrokecolor=:green)
    plot!(Xline, map(x -> f(x, function_name), Xline), label="Truth", color=:green)

    savefig(plt, "./$(experiment)/Plots/$(pipeline_name).pdf")
end