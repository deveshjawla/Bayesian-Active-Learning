using Flux, Plots
using Distributions: Normal, Product
using Statistics
using Distributed
addprocs(8; exeflags=`--project`)
using DelimitedFiles, DataFrames
using LaplaceRedux
PATH = @__DIR__
cd(PATH)

experiment = "DeepEnsembleWithGlorotNormal"
activation_function = "mixed"
function_names = ["Cosine", "Polynomial", "Exponential", "Logarithmic", "sin(pisin)"]#  "Cosine", "Polynomial", "Exponential", "Logarithmic","sin(pisin)"

function nn_without_dropout(input_size, output_size)
    return Chain(
        Parallel(vcat, Dense(input_size => input_size, identity; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal())),
        Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
        Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
        Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
        Dense(4 * input_size => output_size)
    )
end
@everywhere begin
    # load dependencies
    using ProgressMeter
    using CSV

    using Flux
    function network_training(n_epochs, input_size, output_size, data; lr=0.001, dropout_rate=0.2)#::Tuple{Vector{Float32},Any}
        # model = Chain(
        #     Dense(input_size => 5, tanh; init=Flux.glorot_normal()),
        #     # Dropout(dropout_rate),
        #     Dense(5 => 5, tanh; init=Flux.glorot_normal()),
        #     # Dropout(dropout_rate),
        #     Dense(5=> output_size; init=Flux.glorot_normal()),
        # )
        # train_loader = Flux.DataLoader((train_x, train_y), batchsize=batch_size)
        train_loader = data
        model = Chain(
            Parallel(vcat, Dense(input_size => input_size, identity; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal())),
            # Dropout(dropout_rate),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            # Dropout(dropout_rate),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            # Dropout(dropout_rate),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            # Dropout(dropout_rate),
            Dense(4 * input_size => output_size)
        )
        opt = Adam(lr)
        opt_state = Flux.setup(opt, model)

        # @info("Beginning training loop...")
        least_loss = Inf32
        last_improvement = 0
        optim_params = 0
        re = 0

        for epoch_idx in 1:n_epochs
            # global best_acc, last_improvement
            loss = 0.0
            for (x, y) in train_loader
                # Compute the loss and the gradients:
                l, gs = Flux.withgradient(m -> Flux.mse(m(x), y), model)

                if !isfinite(l)
                    # @warn "loss is $l" epoch_idx
                    continue
                end
                begin
                    # Update the model parameters (and the Adam momenta):
                    Flux.update!(opt_state, model, gs[1])
                    # Accumulate the mean loss, just for logging:
                    loss += l / length(train_loader)
                end
            end

            if mod(epoch_idx, 10) == 1
                # Report on train and test, only every 2nd epoch_idx:
                # @info "Loss after epoch_idx $epoch_idx is" loss
            end

            # If this is the minimum loss we've seen so far, save the model out
            if abs(loss) < abs(least_loss)
                # @info(" -> New minimum loss of $loss ! Saving model weights at Epoch $epoch_idx")
                optim_params, re = Flux.destructure(model)
                least_loss = loss
                last_improvement = epoch_idx
            end

            # If we haven't seen improvement in 5 epochs, drop our learning rate:
            if epoch_idx - last_improvement >= 10 && opt_state.layers[1].layers[1].weight.rule.eta > 1e-6
                new_eta = opt_state.layers[1].layers[1].weight.rule.eta / 10.0
                # @warn(" -> Haven't improved in a while, dropping learning rate to $(new_eta)!")
                Flux.adjust!(opt_state; eta=new_eta)
                # After dropping learning rate, give it a few epochs to improve
                last_improvement = epoch_idx
            end

            if epoch_idx - last_improvement >= 50
                # @warn(" -> We're calling this converged.")
                break
            end
        end

        return optim_params
    end
end
using SharedArrays
function parallel_network_training(n_networks, nparameters, n_epochs, input_size, output_size, data)#::Matrix{Float32}
    param_matrices_accumulated = SharedMatrix{Float32}(n_networks, nparameters)
    @showprogress pmap(1:n_networks) do i
        network_weights = network_training(n_epochs, input_size, output_size, data)
        param_matrices_accumulated[i, :] = network_weights
    end
    matrix_weights = convert(Matrix{Float32}, param_matrices_accumulated)
    return matrix_weights
end

function f(x, name::String)
    if name == "Polynomial"
        f = x^4 + 2 * x^3 - 3 * x^2 + x
    elseif name == "Cosine"
        f = cos(x)#sin(π * sin(x))
    elseif name == "sin(pisin)"
        f = sin(π * sin(x))
    elseif name == "Exponential"
        f = exp(x)
    elseif name == "Logarithmic"
        f = log(x)
    end
    return f
end


for function_name in function_names
    begin
        if function_name == "Polynomial"
            # f(x) = x^4 + 2 * x^3 - 3 * x^2 + x 
            xs1 = collect(Float32, -3.36:0.02:-2.9)# .
            xs2 = collect(Float32, -1:0.02:1.8)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -3.6:0.01:2.3)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)
        elseif function_name == "Cosine" || function_name == "sin(pisin)"
            # f(x) = cos(x) 
            xs1 = collect(Float32, -7:0.05:-1)# .
            xs2 = collect(Float32, 3:0.05:5)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -10:0.001:10)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)
        elseif function_name == "Exponential"
            # f(x) = exp(x) 
            xs1 = collect(Float32, -1:0.01:1)# .
            xs2 = collect(Float32, 2:0.01:4.5)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, -2:0.01:5)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)

        elseif function_name == "Logarithmic"
            # f(x) = log(x) 
            xs1 = collect(Float32, 0.01:0.005:1)# .
            xs2 = collect(Float32, 2:0.01:3)# .
            xs = vcat(xs1, xs2)
            Xline = collect(Float32, 0:0.005:6)
            ys = map(x -> f(x, function_name) + rand(Normal(0.0f0, 0.1f0)), xs)

        end
    end

    X = [[i] for i in xs]
    data = zip(X, ys)
    train_x = hcat(X...)
    train_y = permutedims(ys)
    input_size, output_size = 1, 1
    m = nn_without_dropout(input_size, output_size)
    params_, rec = Flux.destructure(m)
    num_params = lastindex(params_)
    ensemble_size = 100
    # trained_params = network_training(1000, input_size, output_size, data)
    trained_params = parallel_network_training(ensemble_size, num_params, 500, input_size, output_size, data)
    # nn = reconstruct(trained_params)

    plt = 0
    # posterior_predictive_mean_samples = []
    # posterior_predictive_full_samples = []

    # @showprogress 1 "Computing LA..." for i = 1:ensemble_size
    #     nn_params = trained_params[i, :]
    #     nn = rec(nn_params)
    #     subset_w = :all
    #     la = Laplace(nn; likelihood=:regression, subset_of_weights=subset_w)
    #     # data = Flux.DataLoader((train_x, train_y), batchsize=1)
    #     fit!(la, data)
    #     optimize_prior!(la)
    #     preds = predict(la, permutedims(Xline), link_approx=:mc)
    # 	means = vcat(preds[1, :]...)
    #     standard_deviations = vcat(preds[2, :]...)
    # 	stds = sqrt.(standard_deviations.^2 .+ la.σ^2)
    # 	predictive_distribution = vec(Normal.(means, stds))
    #     postpred_full_sample = rand(Product(predictive_distribution))
    #     push!(posterior_predictive_mean_samples, mean.(predictive_distribution))
    #     push!(posterior_predictive_full_samples, postpred_full_sample)
    # 	plt = plot(Xline, means, ribbon=stds, fillalpha=0.7, ylim= [-2,2], legend=:none, label="Laplace Approximation", fmt=:pdf, size=(600, 400), dpi=600)
    # end

    # posterior_predictive_mean_samples = hcat(posterior_predictive_mean_samples...)

    # pp_mean = mean(posterior_predictive_mean_samples, dims=2)[:]
    # pp_mean_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_mean_samples, dims=2)[:]
    # pp_mean_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_mean_samples, dims=2)[:]

    # posterior_predictive_full_samples = hcat(posterior_predictive_full_samples...)
    # pp_full_lower = mapslices(x -> quantile(x, 0.05), posterior_predictive_full_samples, dims=2)[:]
    # pp_full_upper = mapslices(x -> quantile(x, 0.95), posterior_predictive_full_samples, dims=2)[:]

    # plt = plot(Xline[:], pp_mean, ribbon=(pp_mean .- pp_full_lower, pp_full_upper .- pp_mean), ylim=[-2, 2], legend=:none, label="Full posterior predictive distribution", fmt=:pdf, size=(600, 400), dpi=600)
    # plot!(Xline[:], pp_mean, ribbon=(pp_mean .- pp_mean_lower, pp_mean_upper .- pp_mean), label="Posterior predictive mean distribution (epistemic uncertainty)")

    
    @time begin
        preds = pred_regression(rec, permutedims(Xline), trained_params)

        means = vec(preds[1, :])
        standard_deviations = 3 .* vec(preds[2, :])

        plt = plot(Xline[:], means, ribbon=standard_deviations, fillalpha=0.7, legend=:none, label="Deep Ensemble", fmt=:pdf, size=(600, 400), dpi=600)
    end

    plot!(Xline, map(x -> f(x, function_name), Xline), label="Truth", color=:green)
    # plot!(Xline, vec(nn(permutedims(Xline))), seriestype=:line, label="MLE Estimate", color=:red)
    scatter!(xs, ys, color=:green, label="Training data", markerstrokecolor=:green)

    mkpath("./$(experiment)")
    writedlm("./$(experiment)/$(function_name)_$(activation_function)_weights.csv", trained_params, ',')
    # writedlm("./$(experiment)/$(function_name)/$(activation_function)_sigmas.csv", sigmas, ',')
    savefig(plt, "./$(experiment)/$(function_name)_$(activation_function).pdf")

end

# lPlot = plot(ch1[:lp], label="Chain 1", title="cos Posterior")
# # plot!(lPlot, ch2[:lp], label="Chain 2")

# sigPlot = plot(ch1[:sigma], label="Chain 1", title="Variance")
# # plot!(sigPlot, ch2[:sigma], label="Chain 2")

# plot(lPlot, sigPlot)
