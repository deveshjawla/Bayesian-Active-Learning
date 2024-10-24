@everywhere begin
    # Define a mutable struct to represent the multi-input model
    mutable struct MultiInputModel
        denseheads::Vector{Dense}
        shared_dense::Dense
    end

    Flux.@functor MultiInputModel

    squared(x) = x * x
    cubed(x) = x * x * x


    # Constructor for the model allowing flexibility for different parameters
    function MultiInputModel(num_categories_list::Vector{Int}, n_features::Int, n_outputs::Int;
        activation_fn=relu)

        categorical_heads = nothing
        if num_categories_list[1] == 0
            categorical_heads = []
        else
            # Create categorical heads: each head corresponds to one categorical feature
            categorical_heads = [Dense(num_categories_list[i] => 1, activation_fn) for i in 1:lastindex(num_categories_list)]
        end

        # Create continuous head (for numerical inputs)
        continuous_head_1 = Dense(n_features => 3, tanh)
        # continuous_head_2 = Dense(n_features => 1, squared)
        # continuous_head_3 = Dense(n_features => 1, cubed)
        continuous_heads = [continuous_head_1]#, continuous_head_2, continuous_head_3]

        dense_heads = [categorical_heads..., continuous_heads...]

        # Create shared dense layers that process the concatenated output from all heads
        total_outputdims_cat = lastindex(categorical_heads)  # Sum of the outputs from all categorical heads
        total_outputdims_cont = lastindex(continuous_heads)  # Sum of the outputs from all categorical heads
        shared_input_dims = total_outputdims_cat + 3 * total_outputdims_cont
        shared_dense = Dense(shared_input_dims, n_outputs, bias=false)

        # Return an instance of MultiInputModel
        return MultiInputModel(dense_heads, shared_dense)
    end

    # Define the forward pass for the model by overloading the function call operator
    function (m::MultiInputModel)(inputs::AbstractArray, cat_indices_list::AbstractArray)
        # Efficiently collect outputs from categorical heads using broadcasting
        categorical_outputs = map((head, indices) -> head(inputs[indices, :]), m.denseheads, cat_indices_list)

        # Efficient concatenation of categorical outputs
        combined_categorical_output = reduce(vcat, categorical_outputs)

        # Pass the combined output through the shared dense layers
        return m.shared_dense(combined_categorical_output)
    end
end


# @everywhere begin
#     # Define a mutable struct to represent the multi-input model
#     mutable struct MultiInputModel
#         categorical_heads::Vector{Dense}
#         continuous_heads::Vector{Dense}
#         shared_dense::Dense
#     end

#     Flux.@functor MultiInputModel

#     squared(x) = x * x
#     cubed(x) = x * x * x

#     # # Constructor for the model allowing flexibility for different parameters
#     # function MultiInputModel(num_categories_list::Vector{Int}, embedding_dims::Vector{Int}, outputdims_cat::Vector{Int}, n_features::Int, outputdims_cont::Int, dense_width::Int, n_outputs::Int;
#     #     activation_fn=relu)

#     #     # Ensure that the input vectors for categorical heads match in size
#     #     if length(num_categories_list) != length(embedding_dims) || length(embedding_dims) != length(outputdims_cat)
#     #         error("Length of num_categories_list, embedding_dims, and outputdims_cat must be the same.")
#     #     end

#     #     # Create categorical heads: each head corresponds to one categorical feature
#     #     categorical_heads = [Chain(Embedding(num_categories_list[i], embedding_dims[i]), Dense(embedding_dims[i], outputdims_cat[i], activation_fn)) for i in 1:length(num_categories_list)]

#     #     # Create continuous head (for numerical inputs)
#     #     continuous_head = Chain(Dense(n_features, outputdims_cont, activation_fn))

#     #     # Create shared dense layers that process the concatenated output from all heads
#     #     total_outputdims_cat = sum(outputdims_cat)  # Sum of the outputs from all categorical heads
#     #     shared_input_dims = total_outputdims_cat + outputdims_cont
#     #     shared_output_dims = round(Int, max(shared_input_dims / 2, 8))
#     #     shared_dense = Chain(Dense(shared_input_dims, shared_output_dims, activation_fn),
#     #         Dense(shared_output_dims, n_outputs, bias=false),  # Final output layer
#     #     )

#     #     # Return an instance of MultiInputModel
#     #     return MultiInputModel(categorical_heads, continuous_head, shared_dense)
#     # end

#     # # Constructor for the model allowing flexibility for different parameters
#     # function MultiInputModel(num_categories_list::Vector{Int}, n_features::Int, n_outputs::Int;
#     #     activation_fn=relu)

#     #     embedding_dims = [round(Int, min(sqrt(x), 50)) for x in num_categories_list]

#     #     outputdims_cat = [round(Int, min(sqrt(x), 7)) for x in embedding_dims]

#     #     # Create categorical heads: each head corresponds to one categorical feature
#     #     categorical_heads = [Chain(Embedding(num_categories_list[i], embedding_dims[i]), Dense(embedding_dims[i], outputdims_cat[i], activation_fn)) for i in 1:length(num_categories_list)]

#     #     outputdims_cont = round(Int, min(sqrt(n_features), 100))

#     #     # Create continuous head (for numerical inputs)
#     #     continuous_head = Chain(Dense(n_features, outputdims_cont, activation_fn))

#     #     # Create shared dense layers that process the concatenated output from all heads
#     #     total_outputdims_cat = sum(outputdims_cat)  # Sum of the outputs from all categorical heads
#     #     shared_input_dims = total_outputdims_cat + outputdims_cont
#     #     shared_output_dims = round(Int, min(sqrt(shared_input_dims), 50))
#     #     shared_dense = Chain(Dense(shared_input_dims, shared_output_dims, activation_fn),
#     #         Dense(shared_output_dims, n_outputs, bias=false),  # Final output layer
#     #     )

#     #     # Return an instance of MultiInputModel
#     #     return MultiInputModel(categorical_heads, continuous_head, shared_dense)
#     # end

#     # Constructor for the model allowing flexibility for different parameters
#     function MultiInputModel(num_categories_list::Vector{Int}, n_features::Int, n_outputs::Int;
#         activation_fn=relu)

#         # Create categorical heads: each head corresponds to one categorical feature
#         categorical_heads = [Dense(num_categories_list[i] => 1, activation_fn) for i in 1:lastindex(num_categories_list)]

#         # Create continuous head (for numerical inputs)
#         continuous_head_1 = Dense(n_features => 1, identity)
#         continuous_head_2 = Dense(n_features => 1, squared)
#         continuous_head_3 = Dense(n_features => 1, cubed)
#         continuous_heads = [continuous_head_1, continuous_head_2, continuous_head_3]

#         # Create shared dense layers that process the concatenated output from all heads
#         total_outputdims_cat = lastindex(num_categories_list)  # Sum of the outputs from all categorical heads
#         total_outputdims_cont = lastindex(continuous_heads)  # Sum of the outputs from all categorical heads
#         shared_input_dims = total_outputdims_cat + total_outputdims_cont
#         shared_dense = Dense(shared_input_dims, n_outputs, bias=false)

#         # Return an instance of MultiInputModel
#         return MultiInputModel(categorical_heads, continuous_heads, shared_dense)
#     end

#     # # Define the forward pass for the model by overloading the function call operator
#     # function (m::MultiInputModel)(categorical_inputs::AbstractArray, numerical_input::AbstractArray)
#     #     # Efficiently collect outputs from categorical heads using broadcasting
#     #     categorical_outputs = map((head, input) -> head(input), m.categorical_heads, categorical_inputs)

#     #     # Efficient concatenation of categorical outputs
#     #     combined_categorical_output = reduce(vcat, categorical_outputs)

#     #     # Apply the continuous head to the numerical input
#     #     continuous_output = m.continuous_head(numerical_input)

#     #     # @show size(combined_categorical_output)
#     #     # @show size(continuous_output)
#     #     # Concatenate categorical and continuous outputs
#     #     combined_output = vcat(combined_categorical_output, continuous_output)

#     #     # Pass the combined output through the shared dense layers
#     #     return m.shared_dense(combined_output)
#     # end

#     # Define the forward pass for the model by overloading the function call operator
#     function (m::MultiInputModel)(inputs::AbstractArray, cat_indices_list::AbstractArray, cont_indices)

#         # Efficiently collect outputs from categorical heads using broadcasting
#         categorical_outputs = map((head, indices) -> head(inputs[indices, :]), m.categorical_heads, cat_indices_list)

#         # Efficient concatenation of categorical outputs
#         combined_categorical_output = reduce(vcat, categorical_outputs)

#         continuous_outputs = map(head -> head(inputs[cont_indices, :]), m.continuous_heads)

#         # Efficient concatenation of categorical outputs
#         combined_continuous_output = reduce(vcat, continuous_outputs)

#         # @show size(combined_categorical_output)
#         # @show size(continuous_output)
#         # Concatenate categorical and continuous outputs
#         combined_output = vcat(combined_categorical_output, combined_continuous_output)

#         # Pass the combined output through the shared dense layers
#         return m.shared_dense(combined_output)
#     end

#     # function (m::MultiInputModel)(inputs::AbstractArray)
#     #     combined_categorical_output = ones(lastindex(m.categorical_heads), size(inputs, 2))  # Preallocate the output vector
#     #     counter = 0
#     #     for (i, cat_layer) in enumerate(m.categorical_heads)
#     #         input_size = size(cat_layer.weight, 2)
#     #         output = cat_layer(inputs[counter+1:counter+input_size, :])  # Use view to avoid copying
#     #         counter += input_size
#     #         combined_categorical_output[i, :] = output  # Directly assign to the preallocated vector
#     #     end

#     #     # categorical_outputs = map((head, input) -> head(input), m.categorical_heads, categorical_inputs)

#     #     # # Efficient concatenation of categorical outputs
#     #     # combined_categorical_output = reduce(vcat, categorical_outputs)

#     #     input_size = size(m.continuous_heads[1].weight, 2)
#     #     # Apply the continuous head to the numerical input

#     #     combined_continuous_output = ones(lastindex(m.continuous_heads), size(inputs, 2))  # Preallocate the output vector
#     #     for (i, cont_layer) in enumerate(m.continuous_heads)
#     #         output = cont_layer(inputs[counter+1:counter+input_size, :]) # Use view to avoid copying
#     #         combined_continuous_output[i, :] = output  # Directly assign to the preallocated vector
#     #     end

#     #     # @show size(combined_categorical_output)
#     #     # @show size(continuous_output)
#     #     # Concatenate categorical and continuous outputs
#     #     combined_output = vcat(combined_categorical_output, combined_continuous_output)

#     #     # Pass the combined output through the shared dense layers
#     #     return m.shared_dense(combined_output)
#     # end
# end
# # # Define loss function: mean squared error for regression
# loss(categorical_input, numerical_input, target) = Flux.mse(model(categorical_input, numerical_input), target)

# # Optimizer
# opt = ADAM()

# # Example target output
# target = [[1, 0] [0, 1] [1, 0] [0, 1] [1, 0]]

# # Sample training data
# data = [(indices_tensor, numerical_input, target)]

# # Train the model
# Flux.train!(loss, Flux.params(model), data, opt)



function mcmc_inference(prior::Tuple, training_data::Tuple{Matrix{Float32},Matrix{Int64}}, n_input, n_output, categorical_indices_list, nsteps::Int, n_chains::Int, al_step::Int, experiment_name::String, pipeline_name::String, mcmc_init_params, temperature, sample_weights, likelihood_name, prior_informativeness, noise_x, noise_y)::Tuple{Array{Float32},Array{Float32,2},Array{Float32,2},Float32}
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
    @everywhere categorical_indices_list = $categorical_indices_list
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))

    # println(eltype(train_x), eltype(train_y))
    # println(mean(location), mean(scale))
    if temperature isa Number && likelihood_name == "TemperedLikelihood"
        @everywhere model = temperedBNN(train_x, train_y, location, scale, temperature)
    elseif likelihood_name == "WeightedLikelihood" && temperature == "CWL"
        @everywhere model = classweightedBNN(train_x, train_y, categorical_indices_list, location, scale, sample_weights)
    else
        @everywhere model = BNN(train_x, train_y, location, scale)
    end

    # if noise_x && !noise_y
    #     model = softmax_bnn_noise_x(train_x,  train_y, location, scale, sample_weights)
    # elseif noise_y && !noise_x
    #     model = softmax_bnn_noise_y(train_x,  train_y, location, scale, sample_weights)
    # elseif noise_y && noise_x
    #     model = softmax_bnn_noise_xy(train_x,  train_y, location, scale, sample_weights)
    # else
    #     model = softmax_bnn(train_x,  train_y, location, scale, sample_weights)
    # end

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

    if noise_x && !noise_y
        noises_x = MCMCChains.group(chains, :noise_x).value #get posterior MCMC samples for network weights
    elseif noise_y && !noise_x
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

    mc_autocor_lag = 2

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
        if noise_x && !noise_y
            noise_set_x = collect.(Float32, eachrow(noises_x[:, :, i]))
            noise_x_matrix = mapreduce(permutedims, vcat, noise_set_x)
        elseif noise_y && !noise_x
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

                if noise_x && !noise_y
                    independent_noise_x_matrix[Int((j) / mc_autocor_lag), :] = noise_x_matrix[j+burn_in, :]
                elseif noise_y && !noise_x
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
        if noise_x && !noise_y
            noise_x_vectors_accumulated[(i-1)*size(independent_noise_x_matrix, 1)+1:i*size(independent_noise_x_matrix, 1), :] = independent_noise_x_matrix
        elseif noise_y && !noise_x
            noise_y_vectors_accumulated[(i-1)*size(independent_noise_y_matrix, 1)+1:i*size(independent_noise_y_matrix, 1), :] = independent_noise_y_matrix
        elseif noise_y && noise_x
            noise_x_vectors_accumulated[(i-1)*size(independent_noise_x_matrix, 1)+1:i*size(independent_noise_x_matrix, 1), :] = independent_noise_x_matrix
            noise_y_vectors_accumulated[(i-1)*size(independent_noise_y_matrix, 1)+1:i*size(independent_noise_y_matrix, 1), :] = independent_noise_y_matrix
        end
    end
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$(al_step)_MAP.csv", mean(map_params_accumulated, dims=1), ',')
    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    if noise_x && !noise_y
        noise_x_vectors_accumulated = collect.(Float32, eachrow(noise_x_vectors_accumulated))
    elseif noise_y && !noise_x
        noise_y_vectors_accumulated = collect.(Float32, eachrow(noise_y_vectors_accumulated))
    elseif noise_y && noise_x
        noise_x_vectors_accumulated = collect.(Float32, eachrow(noise_x_vectors_accumulated))
        noise_y_vectors_accumulated = collect.(Float32, eachrow(noise_y_vectors_accumulated))
    end

    return param_matrices_accumulated, noise_x_vectors_accumulated, noise_y_vectors_accumulated, elapsed
end
