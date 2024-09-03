function mcmc_inference(prior::Tuple, training_data::Tuple{Matrix{Float32}, Matrix{Int64}}, n_input, n_output, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness, noise_x, noise_y)::Tuple
    location, scale = prior
    @everywhere location = $location
    @everywhere scale = $scale
    @everywhere mcmc_init_params = $mcmc_init_params
    # @everywhere network_shape = $network_shape
    nparameters = lastindex(location)
    @everywhere nparameters = $nparameters
    train_x, train_y = training_data
    @everywhere train_x = $train_x
    @everywhere train_y = $train_y
    @everywhere sample_weights = $sample_weights
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # println(eltype(train_x), eltype(train_y))
    # println(mean(location), mean(scale))
    # if temperature isa Number && likelihood_name == "TemperedLikelihood"
    #     @everywhere model = temperedBNN(train_x, train_y, location, scale, temperature)
    # elseif likelihood_name == "WeightedLikelihood" && temperature == "CWL"
    #     @everywhere model = classweightedBNN(train_x, train_y, location, scale, sample_weights)
    # else
    #     @everywhere model = BNN(train_x, train_y, location, scale)
    # end

	if noise_x
		model = softmax_bnn_noise_x(train_x, train_y, location, scale, sample_weights)
	elseif noise_y
		model = softmax_bnn_noise_y(train_x, train_y, location, scale, sample_weights)
	elseif noise_y && noise_x
		model = softmax_bnn_noise_xy(train_x, train_y, location, scale, sample_weights)
	else
		model = softmax_bnn(train_x, train_y, location, scale, sample_weights)
	end

    if al_step == 1 || prior_informativeness == "NoInit"
        chain_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff()), MCMCDistributed(), nsteps, n_chains, progress=false)
    else
        chain_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff()), MCMCDistributed(), nsteps, n_chains, init_params=repeat([mcmc_init_params], n_chains), progress=false)
    end
    chains = chain_timed.value
    elapsed = Float32(chain_timed.time)
    println("It took $(elapsed) seconds to complete the $(nsteps) iterations")
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/elapsed.csv", elapsed)
    θ = MCMCChains.group(chains, :θ).value
    # noise_x = MCMCChains.group(chains, :noise_x).value

	if noise_x
		noises_x = MCMCChains.group(chains, :noise_x).value #get posterior MCMC samples for network weights
	elseif noise_y
		noises_y = MCMCChains.group(chains, :noise_y).value
	elseif noise_y && noise_x
		noises_x = MCMCChains.group(chains, :noise_x).value #get posterior MCMC samples for network weights
		noises_y = MCMCChains.group(chains, :noise_y).value
	end

	# acr_lags = [5, 10, 50, 100]
    # acr = MCMCChains.autocor(chains, lags=acr_lags)
	# autocorrelations_df = DataFrame(acr)
	# means_acr = map(mean, eachcol(autocorrelations_df[:, 2:end]))

	# min_acr = Inf
	# min_acr_idx=1
	# for (i,j) in enumerate(means_acr)
	# 	if j < min_acr && j > 0
	# 		# println(i,j, min_acr)
	# 		min_acr = j
	# 		min_acr_idx = i
	# 	end
	# end

	# mc_autocor_lag = acr_lags[min_acr_idx]
	# @info "Autocorrelation IDX" means_acr

	mc_autocor_lag=2

    burn_in = 0#Int(0.6 * nsteps)
    n_indep_samples = round(Int, (nsteps - burn_in) / mc_autocor_lag)
    # hyperpriors_accumulated = Array{Float32}(undef, n_chains*n_indep_samples, nparameters - lastindex(location))
    param_matrices_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, nparameters)
    noise_x_vectors_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, n_input)
    noise_y_vectors_accumulated = Array{Float32}(undef, n_chains * n_indep_samples, n_output)

    for i in 1:n_chains
        params_set = collect.(eachrow(θ[:, :, i]))
		noise_x_matrix = 0
		noise_y_matrix = 0
		if noise_x
			noise_set_x = collect.(Float32, eachrow(noises_x[:, :, i]))
			noise_x_matrix = mapreduce(permutedims, vcat, noise_set_x)
		elseif noise_y
			noise_set_y = collect.(Float32, eachrow(noises_y[:, :, i]))
        	noise_y_matrix = mapreduce(permutedims, vcat, noise_set_y)
		elseif noise_y && noise_x
			noise_set_x = collect.(Float32, eachrow(noises_x[:, :, i]))
			noise_set_y = collect.(Float32, eachrow(noises_y[:, :, i]))
			noise_x_matrix = mapreduce(permutedims, vcat, noise_set_x)
        	noise_y_matrix = mapreduce(permutedims, vcat, noise_set_y)
		end
        param_matrix = mapreduce(permutedims, vcat, params_set)

        independent_param_matrix = Array{Float32}(undef, n_indep_samples, nparameters)
        independent_noise_x_matrix = Array{Float32}(undef, n_indep_samples, n_input)
        independent_noise_y_matrix = Array{Float32}(undef, n_indep_samples, n_output)
        for j in 1:nsteps-burn_in
            if j % mc_autocor_lag == 0
                independent_param_matrix[Int((j) / mc_autocor_lag), :] = param_matrix[j+burn_in, :]
				
				if noise_x
					independent_noise_x_matrix[Int((j) / mc_autocor_lag), :] = noise_x_matrix[j+burn_in, :]
				elseif noise_y
					independent_noise_y_matrix[Int((j) / mc_autocor_lag), :] = noise_y_matrix[j+burn_in, :]
				elseif noise_y && noise_x
					independent_noise_x_matrix[Int((j) / mc_autocor_lag), :] = noise_x_matrix[j+burn_in, :]
					independent_noise_y_matrix[Int((j) / mc_autocor_lag), :] = noise_y_matrix[j+burn_in, :]
				end
            end
        end

        elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf = convergence_stats(i, chains, elapsed)

        writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "max_psrf"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf]], ',')

        # # lPlot = plot(chains[:lp], title="Log Posterior", label=:none)
        # df = DataFrame(chains)
        # df[!, :chain] = categorical(df.chain)
        # lPlot = Gadfly.plot(df, y=:lp, x=:iteration, Geom.line, color=:chain, Guide.title("Log Posterior"), Coord.cartesian(xmin=df.iteration[1], xmax=df.iteration[1] + nsteps)) # need to change :acceptance_rate to Log Posterior
        # # plt= plot(lPlot, size=(1600, 600))
        # # savefig(plt, "./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/chain_$(i).pdf")
        # lPlot |> PDF("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_lp.pdf", 800pt, 600pt)

        # lp, maxInd = findmax(chains[:lp])
        # params, internals = chains.name_map
        # bestParams = map(x -> chains[x].data[maxInd], params[1:nparameters])
        # map_params_accumulated[i, :] = bestParams
        # println(oob_rhat)

        param_matrices_accumulated[(i-1)*size(independent_param_matrix, 1)+1:i*size(independent_param_matrix, 1), :] = independent_param_matrix
		if noise_x
			noise_x_vectors_accumulated[(i-1)*size(independent_noise_x_matrix, 1)+1:i*size(independent_noise_x_matrix, 1), :] = independent_noise_x_matrix
		elseif noise_y
			noise_y_vectors_accumulated[(i-1)*size(independent_noise_y_matrix, 1)+1:i*size(independent_noise_y_matrix, 1), :] = independent_noise_y_matrix
		elseif noise_y && noise_x
			noise_x_vectors_accumulated[(i-1)*size(independent_noise_x_matrix, 1)+1:i*size(independent_noise_x_matrix, 1), :] = independent_noise_x_matrix
			noise_y_vectors_accumulated[(i-1)*size(independent_noise_y_matrix, 1)+1:i*size(independent_noise_y_matrix, 1), :] = independent_noise_y_matrix
		end
    end
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$(al_step)_MAP.csv", mean(map_params_accumulated, dims=1), ',')
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
	if noise_x
		noise_x_vectors_accumulated = collect.(Float32, eachrow(noise_x_vectors_accumulated))
	elseif noise_y
		noise_y_vectors_accumulated = collect.(Float32, eachrow(noise_y_vectors_accumulated))
	elseif noise_y && noise_x
		noise_x_vectors_accumulated = collect.(Float32, eachrow(noise_x_vectors_accumulated))
		noise_y_vectors_accumulated = collect.(Float32, eachrow(noise_y_vectors_accumulated))
	end
	
    return param_matrices_accumulated, noise_x_vectors_accumulated, noise_y_vectors_accumulated, elapsed
end
