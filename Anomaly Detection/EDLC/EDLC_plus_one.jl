using Distributed
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)
@everywhere begin
    using EvidentialFlux
    using Flux
    using Plots
    PATH = @__DIR__
    cd(PATH)



    # Generate data
    n = 200
    X, y = gen_3_clusters(n)

    input_size = size(X)[1]
    output_size = size(y)[1]

    l1, l2 = 8, 8
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    n_output_layer = l2 * output_size + output_size

    num_params = nl1 + nl2 + n_output_layer

    m = Chain(
            Dense(input_size => 8, relu),
            Dense(8 => 8, relu),
            DIR(8 => output_size)
        )

		_, re = Flux.destructure(m)

    function network_training(input_size, output_size; n_epochs, train_loader, sample_weights_loader)::Vector{Float32}

        # Define model
        m = Chain(
            Dense(input_size => 8, relu),
            Dense(8 => 8, relu),
            DIR(8 => output_size)
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
        epochs = n_epochs
        trnlosses = zeros(epochs)
        for e in 1:epochs
            local trnloss = 0
            grads = Flux.gradient(p) do
                α = m(X)
                # trnloss = Flux.mse(y, α)
                trnloss = dirloss(y, α, e)
                trnloss
            end
            trnlosses[e] = trnloss
            Flux.Optimise.update!(opt, p, grads)
        end

        optim_params, re = Flux.destructure(m)
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
function ensemble_training(num_params::Int, input_size::Int, output_size::Int, acq_size::Int, training_data; ensemble_size::Int=100, sample_weights=nothing, n_epochs=500)::Tuple{Array{Float32,2},Float32}
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

param_matrix, elapsed = ensemble_training(num_params, input_size, output_size, 600, (X, y))

xs = Float32.(-7:0.1:7)
ys = Float32.(-7:0.1:7)
heatmap(xs, ys, (x, y) -> pred_analyzer_multiclass(re, reshape([x, y], (:, 1)), param_matrix)[2]) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./relu_ensemble_edlc_plus_one.pdf")

# scatter(1:epochs, trnlosses, width=80, height=30)

# # Test predictions
# α̂ = m(X)
# ŷ = α̂ ./ sum(α̂, dims=1)
# u = edlc_uncertainty(α̂)

# # Show epistemic uncertainty
# heatmap(-7:0.1:7, -7:0.1:7, (x, y) -> edlc_uncertainty(m(vcat(x, y)))[1])
# scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
# scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
# scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
# savefig("./relu_edlc.pdf")