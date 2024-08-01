using StatsBase
using Flux, Turing
using Turing.Variational

function vi_inference(prior::Tuple, training_data::Tuple{Matrix{Float32},Matrix{Int64}}, n_input, nsteps::Int, al_step::Int, experiment_name::String, pipeline_name::String, temperature, sample_weights, likelihood_name)::Tuple{Array{Float32,2},Vector{Vector{Float32}},Float32}
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
    if temperature isa Number && likelihood_name == "TemperedLikelihood"
        @everywhere model = temperedBNN(train_x, train_y, location, scale, temperature)
    elseif likelihood_name == "WeightedLikelihood" && temperature == "CWL"
        @everywhere model = classweightedBNN(train_x, train_y, location, scale, sample_weights)
    else
        @everywhere model = softmax_bnn_noise_x(train_x, train_y, location, scale)
    end

    q0 = Variational.meanfield(model)
    advi = ADVI(10, 1_000)
    opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
    q = vi(model, advi, q0; optimizer=opt)

    elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf = convergence_stats(i, chains, elapsed)

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/$(al_step)_chain_$i.csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess", "max_psrf"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess, max_psrf]], ',')

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/posterior_dist/$al_step.csv", [q.dist.m q.dist.σ], ',')

    return param_matrices_accumulated, noise_x_vectors_accumulated, elapsed
end