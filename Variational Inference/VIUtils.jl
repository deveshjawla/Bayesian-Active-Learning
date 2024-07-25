using StatsBase

function vi_inference(prior::Tuple, training_data::Tuple{Array{Float32, 2}, Array{Int, 2}}, nsteps::Int, n_epochs::Int, al_step::Int, experiment_name::String, pipeline_name::String)::Tuple{Array{Float32, 2}, Float32}
	sigma, nparameters = prior
	# @everywhere network_shape = $network_shape
	@everywhere nparameters = $nparameters
	train_x, train_y = training_data
	@everywhere train_x = $train_x
	@everywhere train_y = $train_y
	println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
	# println(eltype(train_x), eltype(train_y))
	@everywhere model = bayesnnMVG(train_x, train_y, sigma, nparameters)
	q = Variational.meanfield(model)

	μ = randn(lastindex(q))
	ω = exp.(-1 .* ones(lastindex(q)))

	q = AdvancedVI.update(q, μ, ω)

	# @info("Beginning training loop...")
	best_elbo = -Inf32
	last_improvement = 0
	q̂ = 0
	elapsed = 0
	param_matrices_accumulated = 0
	for epoch_idx = 1:n_epochs
		advi = ADVI(10, nsteps)
		chain_timed = @timed vi(model, advi, q)
		q_hat = chain_timed.value
		elapsed = Float32(chain_timed.time)
		elbo_score = elbo(advi, q, model, 1000)
		
		# If this is the minimum loss we've seen so far, save the model out
		if elbo_score > best_elbo
			@info(" -> New best ELBO! Saving model weights")
			q̂ = q_hat
			best_elbo = elbo_score
			last_improvement = epoch_idx
			param_matrices_accumulated = permutedims(rand(q_hat, 5000))
		end

		if epoch_idx - last_improvement >= 5
			@warn(" -> We're calling this converged.")
			break
		end
	end
	# writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/elapsed.csv", elapsed)
	
	writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step).csv", [["elapsed", "ELBO"] [elapsed, best_elbo]], ',')
		# println(oob_rhat)

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/posterior_dist/$al_step.csv", [q̂.dist.m q̂.dist.σ], ',')
	return param_matrices_accumulated, elapsed
end