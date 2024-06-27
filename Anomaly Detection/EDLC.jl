using Distributed
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)
@everywhere begin
    using EvidentialFlux
    using Flux
    using Plots
    PATH = @__DIR__
    cd(PATH)

    function gendata(n)
        x1 = randn(Float32, 2, n)
        x2 = randn(Float32, 2, n) .+ [2, 2]
        x3 = randn(Float32, 2, n) .+ [-2, 2]
        y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
        y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
        y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
        hcat(x1, x2, x3), permutedims(hcat(y1, y2, y3))
    end

    # Generate data
    n = 200
    X, y = gendata(n)

    input_size = size(X)[1]
    output_size = size(y)[1]

    l1, l2 = 8, 8
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    n_output_layer = l2 * output_size #+ output_size

    num_params = nl1 + nl2 + n_output_layer

    m = Chain(
        Dense(input_size => 8, relu; bias=true),
        Dense(8 => 8, relu; bias=true),
        DIR(8 => output_size; bias=false)
    )

    _, re = Flux.destructure(m)

    function network_training(input_size, output_size; n_epochs, train_loader, sample_weights_loader)::Vector{Float32}

        # Define model
        m = Chain(
            Dense(input_size => 8, relu; bias=true),
            Dense(8 => 8, relu; bias=true),
            DIR(8 => output_size; bias=false)
        )

        # m = Chain(
        #     Parallel(vcat,
        #         Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
        #         Dense(input_size => input_size, relu; init=Flux.glorot_normal()),
        #         Dense(input_size => input_size, tanh; init=Flux.glorot_normal()),
        #         Dense(input_size => input_size, sin; init=Flux.glorot_normal())),
        # 		Parallel(vcat,
        #         Dense(4*input_size => input_size, relu; init=Flux.glorot_normal()),
        #         Dense(4*input_size => input_size, relu; init=Flux.glorot_normal()),
        #         Dense(4*input_size => input_size, tanh; init=Flux.glorot_normal()),
        #         Dense(4*input_size => input_size, sin; init=Flux.glorot_normal())),
        #     DIR(4 * input_size => output_size)
        # )
        opt = Flux.Optimise.AdaBelief()
        p = Flux.params(m)

        # Train it
        least_loss = Inf32
        last_improvement = 0
        optim_params = 0

        epochs = n_epochs
        trnlosses = zeros(epochs)
        for e in 1:epochs
            # local trnloss = 0
            # grads = Flux.gradient(p) do
            #     α = m(x)
            #     # trnloss = Flux.mse(y, α)
            #     trnloss = dirloss(y, α, e)
            #     trnloss
            # end
            # trnlosses[e] = trnloss
            # Flux.Optimise.update!(opt, p, grads)

            loss = 0.0
            for (x, y) in train_loader
                # Compute the loss and the gradients:
                local l = 0.0
                grads = Flux.gradient(p) do
                    α = m(x)
                    # trnloss = Flux.mse(y, α)
                    l = dirloss(y, α, e)
                    l
                end

                # Update the model parameters (and the Adam momenta):
                Flux.Optimise.update!(opt, p, grads)
                # Accumulate the mean loss, just for logging:
                loss += l / length(train_loader)
            end
            trnlosses[e] = loss

            if loss < least_loss
                # @info(" -> New minimum loss! Saving model weights")
                optim_params, re = Flux.destructure(m)
                least_loss = loss
                last_improvement = e
            end

            # #If we haven't seen improvement in 5 epochs, drop our learning rate:
            # if e - last_improvement >= 10 && opt.eta > 1e-6
            #     new_eta = opt.eta / 10.0
            #     # @warn(" -> Haven't improved in a while, dropping learning rate to $(new_eta)!")
            #     opt = Flux.Optimise.AdaBelief(new_eta)
            #     # After dropping learning rate, give it a few epochs to improve
            #     last_improvement = e
            # end

            # if e - last_improvement >= 30
            #     # @warn(" -> We're calling this converged.")
            #     break
            # end

        end
        scatter(1:epochs, trnlosses, width=80, height=30)
        savefig("./loss.pdf")

        # optim_params, re = Flux.destructure(m)
        return optim_params
    end

    using SharedArrays
    using ProgressMeter
end
function parallel_network_training(input_size, output_size, n_networks, nparameters, n_epochs, train_loader, sample_weights_loader)::Matrix{Float32}
    param_matrices_accumulated = SharedMatrix{Float32}(n_networks, nparameters)
    @showprogress "Parallel Networks Training Progress" pmap(1:n_networks) do i #@showprogress
        network_weights = network_training(input_size, output_size; n_epochs, train_loader, sample_weights_loader)
        param_matrices_accumulated[i, :] = network_weights
    end
    return convert(Matrix{Float32}, param_matrices_accumulated)
end
function ensemble_training(num_params::Int, input_size::Int, output_size::Int, acq_size::Int, training_data; ensemble_size::Int=100, sample_weights=nothing, n_epochs=100)::Tuple{Array{Float32,2},Float32}
    train_x, train_y = training_data

    # train_y = Flux.onehotbatch(vec(train_y), 1:output_size)
    # println("Checking dimensions of train_x and train_y just before training:", size(train_x), " & ", size(train_y))
    # # println(eltype(train_x), eltype(train_y))
    train_loader = Flux.DataLoader((train_x, train_y), batchsize=acq_size)
    if !isnothing(sample_weights)
        sample_weights_loader = Flux.DataLoader(permutedims(sample_weights), batchsize=acq_size)
    else
        sample_weights_loader = nothing
    end

    chain_timed = @timed parallel_network_training(input_size, output_size, ensemble_size, num_params, n_epochs, train_loader, sample_weights_loader)

    param_matrices_accumulated = convert(Array{Float32,2}, chain_timed.value)
    elapsed = Float32(chain_timed.time)

    # writedlm("./Experiments/$(experiment_name)/$(pipeline_name)/independent_param_matrix_all_chains/$al_step.csv", param_matrices_accumulated, ',')
    return param_matrices_accumulated, elapsed
end

using Statistics
using StatsBase: countmap
function majority_voting(predictions::AbstractVector)::Vector{Float32}
    count_map = countmap(predictions)
    # println(count_map)
    uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
    index_max = argmax(nUniques)
    prediction = uniques[index_max]
    pred_probability = maximum(nUniques) / sum(nUniques)
    return [prediction, 1 - pred_probability] # 1 -  => give the unceratinty
end

uncertainty(α) = first(size(α)) ./ sum(α, dims=1)

function normalized_entropy(prob_vec::Vector, n_output)
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum(prob_vec) < 0.99
        return error("sum(prob_vec) is not 1 BUT $(sum(prob_vec)) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log2.(prob_vec))) / log2(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::Matrix, n_output)
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    return H
end

"""
returns H, E_H, H + E_H
"""
function bald(prob_matrix::Matrix, n_output)
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H + E_H
end

function pred_analyzer_multiclass(reconstruct, test_xs::Array{Float32,2}, params_set::Array{Float32,2})::Array{Float32,2}
    nets = map(reconstruct, eachrow(params_set))
    predictions_nets = map(x -> x(test_xs), nets)
    # predictions_nets = map(x -> x .+ 1, predictions_nets)
    ŷ_prob = map(x -> mapslices(y -> y ./ sum(y), x, dims=1), predictions_nets) #ŷ
    u = mapreduce(x -> mapreduce(uncertainty, hcat, eachcol(x)), vcat, predictions_nets)

    # pred_matrix = cat(predictions_nets..., dims=3)
    # bald_scores = mapslices(x -> bald(x, first(size(x))), pred_matrix, dims=[1, 3])
    # aleatoric_uncertainties = mapreduce(x -> x[2], hcat, bald_scores[1, :, 1])

    ŷ_label = mapreduce(x -> mapreduce(argmax, hcat, eachcol(x)), vcat, ŷ_prob)
    pred_plus_std = mapslices(majority_voting, ŷ_label, dims=1)
    u_plus_std = mapslices(x -> [mean(x), std(x)], u, dims=1)
    pred_matrix = vcat(pred_plus_std, u_plus_std)
    return pred_matrix[[1, 3], :]
end

# function pred_analyzer_multiclass(test_xs::Array{Float32, 2}, params_set::Array{Float32, 2})::Array{Float32, 2}
# 	nets = map(feedforward, eachrow(params_set))
# 	predictions_nets = map(x-> x(test_xs), nets)
# 	ensembles = mapreduce(x-> mapslices(argmax, x, dims=1), vcat, predictions_nets)
# 	pred_matrix = mapslices(majority_voting, ensembles, dims =1)
#     return pred_matrix
# end


param_matrix, elapsed = ensemble_training(num_params, input_size, output_size, 100, (X, y); ensemble_size=8, n_epochs=500)
# pred_analyzer_multiclass(re, X, param_matrix)

xs = Float32.(-10:0.1:10)
ys = Float32.(-10:0.1:10)
heatmap(xs, ys, (x, y) -> pred_analyzer_multiclass(re, reshape([x, y], (:, 1)), param_matrix)[2]) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./EDLC_relu_ensemble.pdf")

# scatter(1:epochs, trnlosses, width=80, height=30)

# # Test predictions
# α̂ = m(X)
# ŷ = α̂ ./ sum(α̂, dims=1)
# u = uncertainty(α̂)

# # Show epistemic uncertainty
# heatmap(-7:0.1:7, -7:0.1:7, (x, y) -> uncertainty(m(vcat(x, y)))[1])
# scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
# scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
# scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
# savefig("./relu_edlc.pdf")