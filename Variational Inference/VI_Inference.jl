
function vi_inference(prior::Tuple, training_data::Tuple{Matrix{Float32},Matrix{Int64}}, al_step::Int, experiment_name::String, pipeline_name::String, temperature, sample_weights, likelihood_name)::Tuple{Array{Float32,2},Vector{Vector{Float32}},Float32}
    location, scale = prior
    # scale = Float32.(scale)
    # location = Float32.(location)
    # nparameters = lastindex(location)
    train_x, train_y = training_data
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))

    # @info eltype(train_x), eltype(train_y)
    if temperature isa Number && likelihood_name == "TemperedLikelihood"
        model = temperedBNN(train_x, train_y, location, scale, temperature)
    elseif likelihood_name == "WeightedLikelihood" && temperature == "CWL"
        model = classweightedBNN(train_x, train_y, location, scale, sample_weights)
    else
        model = BNN(train_x, train_y, location, scale)
    end

    q0 = Variational.meanfield(model)
    advi = ADVI(10, 10000; adtype=AutoReverseDiff())
    opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
    timed_q = @timed vi(model, advi, q0; optimizer=opt)
    q = timed_q.value
    elapsed = Float32(timed_q.time)
    z = rand(q, 1000)

    _, sym2range = bijector(model, Val(true))

    params_set = Float32.(z[first(sym2range.θ), :])
    param_matrix = permutedims(params_set)
    noise_set_x = collect(eachcol(Float32.(z[first(sym2range.noise_x), :])))

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/elapsed_$(al_step).csv", [elapsed], ',')

    writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/convergence_statistics/posterior_dist_$(al_step).csv", [q.dist.m q.dist.σ], ',')

    return param_matrix, noise_set_x, elapsed
end