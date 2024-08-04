
function inference(training_data::Tuple{Matrix{Float32},Any}, al_step::Int, experiment_name::String, pipeline_name::String, learning_algorithm::String)::Tuple{Vector{Float32}, Any,Float32}
    train_x, train_y = training_data
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
	input_size = size(train_x, 1)
	output_size = size(train_y, 1)
	nn_arch_name = 0
	loss_func = 0
	if learning_algorithm == "LaplaceApprox"
		nn_arch_name = "Relu2Layers"
		loss_func = Flux.logitcrossentropy
	elseif learning_algorithm == "Evidential"
		nn_arch_name = "Evidential Classification"
		loss_func = dirloss
	end

    timed_tuple = @timed network_training(nn_arch_name, input_size, output_size, 100; data=(train_x, train_y), loss_function=loss_func)

    optim_theta, re = timed_tuple.value
    elapsed = Float32(timed_tuple.time)

	if learning_algorithm == "LaplaceApprox"
		m=re(optim_theta)
		la = Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
		fit!(la, zip(collect(eachcol(train_x)), train_y))

		#Using Empirical Bayes, optimise the Prior
		optimize_prior!(la; verbose=false, n_steps=100)

		re = la
		@info "Finished LaplaceApprox"
	end

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/elapsed_$(al_step).csv", [elapsed], ',')

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/optim_theta_$(al_step).csv", optim_theta, ',')

    return optim_theta, re, elapsed
end