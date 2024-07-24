# using EvidentialFlux
using Flux
using Distributions
using Turing
using DelimitedFiles
using Statistics
using DataFrames
PATH = @__DIR__
cd(PATH)
include("./../../DirichletLayer.jl")
using Random
list_noise_x = [false, true]
list_noise_y = [false, true]
output_activation_functions = ["Softmax", "Softplus"]
uncertainty_metrics = ["Prediction Probability", "Std of Prediction Probability", "Aleatoric Uncertainty", "Epistemic Uncertainty", "Total Uncertainty"]
number_of_clusters = [1, 3]

stats_matrix = Array{Float32}(undef, 0, 7)
for noise_x in list_noise_x
    for noise_y in list_noise_y
        for num_clusters in number_of_clusters
            # Generate data
            n = 20
            include("./../../DataUtils.jl")
			if num_clusters == 3
				X, y = gen_3_clusters(n)
				Y = argmax.(eachcol(y))
				# Y = y #for one hot labels
				test_X, test_y = gen_3_clusters(100)
            	test_Y = argmax.(eachcol(test_y))
			elseif num_clusters == 1
				X, y = gen_1_clusters(n)
				Y = y
				test_X, test_y = gen_1_clusters(100)
            	test_Y = test_y
			end

            input_size = size(X)[1]
            output_size = size(y)[1]

            l1, l2 = 8, 8
            nl1 = input_size * l1 + l1
            nl2 = l1 * l2 + l2
            n_output_layer = l2 * output_size

            num_params = nl1 + nl2 + n_output_layer
            for output_activation_function in output_activation_function
                if output_activation_function == "Softmax"
                    function feedforward(θ::AbstractVector)
                        W0 = reshape(θ[1:16], 8, 2)
                        b0 = θ[17:24]
                        W1 = reshape(θ[25:88], 8, 8)
                        b1 = θ[89:96]
                        W2 = reshape(θ[97:120], 3, 8)

                        model = Chain(
                            Dense(W0, b0, relu),
                            Dense(W1, b1, relu),
                            Dense(W2, false)
                        )
                        return model
                    end
                elseif output_activation_function == "Softplus"
                    function feedforward(θ::AbstractVector)
                        W0 = reshape(θ[1:16], 8, 2)
                        b0 = θ[17:24]
                        W1 = reshape(θ[25:88], 8, 8)
                        b1 = θ[89:96]
                        W2 = reshape(θ[97:120], 3, 8)

                        model = Chain(
                            Dense(W0, b0, relu),
                            Dense(W1, b1, relu),
                            DIR(W2, false, relu),
                        )
                        return model
                    end
                end


                prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1),
                    sqrt(2 / (l1 + l2)) * ones(nl2),
                    sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))

                num_params = lastindex(prior_std)
                using ReverseDiff

                include("./../../BayesianModel.jl")
                if output_activation_function == "Softmax"
                    if noise_x
                        model = softmax_bnn_noise_x(X, Y, num_params, prior_std)
                    elseif noise_y
                        model = softmax_bnn_noise_y(X, Y, num_params, prior_std)
                    elseif noise_y && noise_x
                        model = softmax_bnn_noise_xy(X, Y, num_params, prior_std)
                    else
                        model = softmax_bnn(X, Y, num_params, prior_std)
                    end
                elseif output_activation_function == "Softplus"
                    if noise_x
                        model = softplus_bnn_noise_x(X, Y, num_params, prior_std)
                    elseif noise_y
                        model = softplus_bnn_noise_y(X, Y, num_params, prior_std)
                    elseif noise_y && noise_x
                        model = softplus_bnn_noise_xy(X, Y, num_params, prior_std)
                    else
                        model = softplus_bnn(X, Y, num_params, prior_std)
                    end
                end

                N = 1000
                ch1_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff()), N)
                ch1 = ch1_timed.value
                elapsed = Float32(ch1_timed.time)

                weights = MCMCChains.group(ch1, :θ).value #get posterior MCMC samples for network weights
                params_set = collect.(Float32, eachrow(weights[:, :, 1]))
                param_matrix = mapreduce(permutedims, vcat, params_set)

                if noise_x
                    noises = MCMCChains.group(ch1, :noise).value #get posterior MCMC samples for network weights
                    noise_set = collect.(Float32, eachrow(noises[:, :, 1]))
                    noise_matrix = mapreduce(permutedims, vcat, noise_set)
                elseif noise_y
                    noises_y = MCMCChains.group(ch1, :noise_y).value
                    noise_set_y = collect.(Float32, eachrow(noises_y[:, :, 1]))
                    noise_matrix_y = mapreduce(permutedims, vcat, noise_set_y)
                elseif noise_y && noise_x
                    noises = MCMCChains.group(ch1, :noise).value #get posterior MCMC samples for network weights
                    noises_y = MCMCChains.group(ch1, :noise_y).value
                    noise_set = collect.(Float32, eachrow(noises[:, :, 1]))
                    noise_matrix = mapreduce(permutedims, vcat, noise_set)

                    noise_set_y = collect.(Float32, eachrow(noises_y[:, :, 1]))
                    noise_matrix_y = mapreduce(permutedims, vcat, noise_set_y)
                end

                using Distributed
                include("./../../MCMCUtils.jl")
                include("./../../ScoringFunctions.jl")
                if noise_x
                    ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set=noise_set; output_activation_function=output_activation_function)
                elseif noise_y
                    ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set_y=noise_set_y; output_activation_function=output_activation_function)
                elseif noise_y && noise_x
                    ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set=noise_set, noise_set_y=noise_set_y; output_activation_function=output_activation_function)
                else
                    ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix; output_activation_function=output_activation_function)
                end

                ŷ = ŷ_uncertainties[1, :]
                acc, f1 = performance_stats_multiclass(ŷ, test_Y)
                @info "Balanced Accuracy and F1 are " acc f1

                stats_matrix = vcat(stats_matrix, [num_clusters output_activation_function noise_x noise_y acc f1 elapsed])

                # test = pred_analyzer_multiclass(reshape([0.0f0,0.0f0], (:, 1)), param_matrix, noise_set = noise_set, noise_set_y = noise_set_y)

                using StatsPlots
                X1 = Float32.(-4:0.1:4)
                X2 = Float32.(-3:0.1:5)

                function pairs_to_matrix(X1, X2)
                    n_pairs = lastindex(X1) * lastindex(X2)
                    test_x_area = Matrix{Float32}(undef, 2, n_pairs)
                    count = 1
                    for x1 in X1
                        for x2 in X2
                            test_x_area[:, count] = [x1, x2]
                            count += 1
                        end
                    end
                    return test_x_area
                end

                test_x_area = pairs_to_matrix(X1, X2)

                if noise_x
                    ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set=noise_set, output_activation_function=output_activation_function)
                elseif noise_y
                    ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set_y=noise_set_y, output_activation_function=output_activation_function)
                elseif noise_y && noise_x
                    ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set=noise_set, noise_set_y=noise_set_y, output_activation_function=output_activation_function)
                else
                    ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, output_activation_function=output_activation_function)
                end

                mkpath("./NumClusters=$(num_clusters)/$(output_activation_function)/noise_x=$(noise)_noise_y=$(noise_y)")

                for (i, j) in enumerate(uncertainty_metrics)

                    uncertainties = reshape(ŷ_uncertainties[i, :], (lastindex(X1), lastindex(X2)))

                    gr(size=(650, 600), dpi=600)
                    heatmap(X1, X2, uncertainties)
                    scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
                    scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
                    scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")

                    savefig("./NumClusters=$(num_clusters)/$(output_activation_function)/noise_x=$(noise)_noise_y=$(noise_y)/$(j)_$(elapsed)_sec.pdf")
                end
            end
        end
    end
end

df = DataFrame(stats_matrix, [:NumClusters, :OutputActivationFunction, :NoiseX, :NoiseY, :WeightedAccuracy, :WeightedF1, :Elapsed])
CSV.write("./stats.csv", df)