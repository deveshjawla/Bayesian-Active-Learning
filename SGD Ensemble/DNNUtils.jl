
using Distributed

# # instantiate and precompile environment in all processes
# @everywhere begin
#     using Pkg
#     Pkg.activate(@__DIR__)
#     Pkg.instantiate()
#     Pkg.precompile()
# end

@everywhere begin
    # load dependencies
    using ProgressMeter
    using CSV
    using Flux
end

using SharedArrays
include("./../AdaBeliefCosAnnealNNTraining.jl")
function parallel_network_training(n_networks, nparameters, n_epochs, n_input, n_output, train_loader, sample_weights_loader)::Matrix{Float32}
    param_matrices_accumulated = SharedMatrix{Float32}(n_networks, nparameters)
    @showprogress "Parallel Networks Training Progress" pmap(1:n_networks) do i #@showprogress
		optim_theta, re = network_training(m, n_input, n_output, n_epochs, train_loader, sample_weights_loader)
        param_matrices_accumulated[i, :] = optim_theta
    end
    return convert(Matrix{Float32}, param_matrices_accumulated)
end

function ensemble_training(num_params::Int, n_input::Int, n_output::Int, acq_size::Int, training_data::Tuple{Array{Float32,2},Array{Int,2}}, ensemble_size::Int, sample_weights; n_epochs=100)::Tuple{Array{Float32,2},Float32}
    train_x, train_y = training_data

    train_y = Flux.onehotbatch(vec(train_y), 1:n_output)
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # println(eltype(train_x), eltype(train_y))
    train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)
    sample_weights_loader = Flux.DataLoader(permutedims(sample_weights), batchsize=acq_size)

    chain_timed = @timed parallel_network_training(ensemble_size, num_params, n_epochs, n_input, n_output, train_loader, sample_weights_loader)

    param_matrices_accumulated = convert(Array{Float32,2}, chain_timed.value)
    elapsed = Float32(chain_timed.time)

    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    return param_matrices_accumulated, elapsed
end