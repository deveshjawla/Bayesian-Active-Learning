
function inference(training_data::Tuple{Matrix{Float32},Any}, al_step::Int, experiment_name::String, pipeline_name::String, learning_algorithm::String)::Tuple{Vector{Float32},Any,Float32}
    train_x, train_y = training_data
    println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    n_input = size(train_x, 1)
    n_output = size(train_y, 1)
    nn_arch_name = 0
    loss_func = 0
    if learning_algorithm == "LaplaceApprox"
        nn_arch_name = "Relu2Layers"
        loss_func = Flux.logitcrossentropy
    elseif learning_algorithm == "Evidential"
        nn_arch_name = "Evidential Classification"
        loss_func = dirloss
    end

    balance_of_training_data = countmap(Int.(train_y))
    sample_weights = similar(train_y, Float32)
    nos_training = lastindex(train_y)
    for i = 1:nos_training
        sample_weights[i] = nos_training / balance_of_training_data[train_y[i]]
    end
    sample_weights ./= n_output

    timed_tuple = @timed network_training(nn_arch_name, n_input, n_output, 100; data=(train_x, train_y), data_weights=sample_weights, loss_function=loss_func)

    optim_theta, re = timed_tuple.value
    elapsed = Float32(timed_tuple.time)

    if learning_algorithm == "LaplaceApprox"
        m = re(optim_theta)
        la = LaplaceRedux.Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
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