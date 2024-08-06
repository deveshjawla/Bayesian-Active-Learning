# using EvidentialFlux
using Flux
using Distributions
using Turing
using DelimitedFiles
using Statistics
using DataFrames
using LazyArrays
using FillArrays
using CSV
PATH = @__DIR__
cd(PATH)
include("./../Evidential Deep Learning/DirichletLayer.jl")
using Random
list_noise_x = [false, true]
list_noise_y = [false, true]
output_activation_functions = ["Relu", "Softmax"]
uncertainty_metrics = ["Predicted Label", "Prediction Probability", "Std of Prediction Probability", "Aleatoric Uncertainty", "Epistemic Uncertainty", "Total Uncertainty"]
number_of_clusters = [3]
list_compile_reversediff = [true, false]
N = 1000 #number of MCMC steps
list_training_sizes = [20, 100]
# n = 20 #number of training samples
let
    stats_matrix = Array{Float32}(undef, 0, 8)
    for compile_reversediff in list_compile_reversediff
        for num_clusters in number_of_clusters
            for n in list_training_sizes
                for output_activation_function in output_activation_functions
                    for noise_x in list_noise_x
                        for noise_y in list_noise_y
                            mkpath("./Experiments/Compile_ReverseDiff=$(compile_reversediff)/NumClusters=$(num_clusters)/Training_Size=$(n)/$(output_activation_function)/noise_x=$(noise_x)_noise_y=$(noise_y)")
                            # Generate data
                            include("./../DataUtils.jl")
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

                            n_input = size(X, 1)
                            n_output = size(y, 1)

                            l1, l2 = 8, 8
                            nl1 = n_input * l1 + l1
                            nl2 = l1 * l2 + l2
                            n_output_layer = l2 * n_output

                            num_params = nl1 + nl2 + n_output_layer

                            if output_activation_function == "Softmax" && n_output == 1
                                include("SoftmaxNetwork1.jl")
                            elseif output_activation_function == "Relu" && n_output == 1
                                include("ReluNetwork1.jl")
                            end

                            if output_activation_function == "Softmax" && n_output == 3
                                include("SoftmaxNetwork3.jl")
                            elseif output_activation_function == "Relu" && n_output == 3
                                include("ReluNetwork3.jl")
                            end


                            prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (n_input + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + n_output)) * ones(n_output_layer)))

                            num_params = lastindex(prior_std)
                            using ReverseDiff

                            include("./../BayesianModel.jl")
                            if compile_reversediff
                                if output_activation_function == "Softmax"
                                    if noise_x
                                        model = fast_softmax_bnn_noise_x(X, Y, num_params, prior_std)
                                    elseif noise_y
                                        model = fast_softmax_bnn_noise_y(X, Y, num_params, prior_std)
                                    elseif noise_y && noise_x
                                        model = fast_softmax_bnn_noise_xy(X, Y, num_params, prior_std)
                                    else
                                        model = fast_softmax_bnn(X, Y, num_params, prior_std)
                                    end
                                elseif output_activation_function == "Relu"
                                    if noise_x
                                        model = fast_softplus_bnn_noise_x(X, Y, num_params, prior_std)
                                    elseif noise_y
                                        model = fast_softplus_bnn_noise_y(X, Y, num_params, prior_std)
                                    elseif noise_y && noise_x
                                        model = fast_softplus_bnn_noise_xy(X, Y, num_params, prior_std)
                                    else
                                        model = fast_softplus_bnn(X, Y, num_params, prior_std)
                                    end
                                end
                            else
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
                                elseif output_activation_function == "Relu"
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
                            end

                            ch1_timed = @timed sample(model, NUTS(; adtype=AutoReverseDiff(; compile=compile_reversediff)), N)
                            ch1 = ch1_timed.value
                            elapsed = Float32(ch1_timed.time)

                            weights = MCMCChains.group(ch1, :θ).value #get posterior MCMC samples for network weights
                            params_set = collect.(Float32, eachrow(weights[:, :, 1]))
                            param_matrix = mapreduce(permutedims, vcat, params_set)

                            if noise_x
                                noises = MCMCChains.group(ch1, :noise_x).value #get posterior MCMC samples for network weights
                                noise_set_x = collect.(Float32, eachrow(noises[:, :, 1]))
                                noise_matrix = mapreduce(permutedims, vcat, noise_set_x)
                            elseif noise_y
                                noises_y = MCMCChains.group(ch1, :noise_y).value
                                noise_set_y = collect.(Float32, eachrow(noises_y[:, :, 1]))
                                noise_matrix_y = mapreduce(permutedims, vcat, noise_set_y)
                            elseif noise_y && noise_x
                                noises = MCMCChains.group(ch1, :noise_x).value #get posterior MCMC samples for network weights
                                noises_y = MCMCChains.group(ch1, :noise_y).value
                                noise_set_x = collect.(Float32, eachrow(noises[:, :, 1]))
                                noise_matrix = mapreduce(permutedims, vcat, noise_set_x)

                                noise_set_y = collect.(Float32, eachrow(noises_y[:, :, 1]))
                                noise_matrix_y = mapreduce(permutedims, vcat, noise_set_y)
                            end

                            using Distributed
                            include("./../MCMCUtils.jl")
                            include("./../ScoringFunctions.jl")
                            if noise_x
                                ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set_x=noise_set_x; output_activation_function=output_activation_function)
                            elseif noise_y
                                ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set_y=noise_set_y; output_activation_function=output_activation_function)
                            elseif noise_y && noise_x
                                ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix, noise_set_x=noise_set_x, noise_set_y=noise_set_y; output_activation_function=output_activation_function)
                            else
                                ŷ_uncertainties = pred_analyzer_multiclass(test_X, param_matrix; output_activation_function=output_activation_function)
                            end

                            ŷ = ŷ_uncertainties[1, :]
                            acc, f1 = performance_stats_multiclass(test_Y, ŷ, n_output)
                            @info "Balanced Accuracy and F1 are " acc f1

                            stats_matrix = vcat(stats_matrix, [compile_reversediff num_clusters n output_activation_function noise_x noise_y acc f1 elapsed])

                            # test = pred_analyzer_multiclass(reshape([0.0f0,0.0f0], (:, 1)), param_matrix, noise_set_x = noise_set_x, noise_set_y = noise_set_y)

                            using StatsPlots
                            X1 = Float32.(-4:0.1:4)
                            X2 = Float32.(-3:0.1:5)

                            test_x_area = pairs_to_matrix(X1, X2)

                            if noise_x
                                ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set_x=noise_set_x, output_activation_function=output_activation_function)
                            elseif noise_y
                                ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set_y=noise_set_y, output_activation_function=output_activation_function)
                            elseif noise_y && noise_x
                                ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, noise_set_x=noise_set_x, noise_set_y=noise_set_y, output_activation_function=output_activation_function)
                            else
                                ŷ_uncertainties = pred_analyzer_multiclass(test_x_area, param_matrix, output_activation_function=output_activation_function)
                            end

                            for (i, j) in enumerate(uncertainty_metrics)

                                uncertainties = reshape(ŷ_uncertainties[i, :], (lastindex(X1), lastindex(X2)))

                                gr(size=(700, 600), dpi=300)
                                heatmap(X1, X2, uncertainties)
                                scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
                                scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
                                scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3", legend_title="Classes", title="Training Size=$(n), $(output_activation_function), ε_input=$(noise_x), ε_output=$(noise_y)", aspect_ratio=:equal, xlim=[-4, 4], ylim=[-3, 5], colorbar_title=" \n$(j)")

                                savefig("./Experiments/Compile_ReverseDiff=$(compile_reversediff)/NumClusters=$(num_clusters)/Training_Size=$(n)/$(output_activation_function)/noise_x=$(noise_x)_noise_y=$(noise_y)/$(j).pdf")
                            end
                        end
                    end
                end
            end
        end
    end

    df = DataFrame(stats_matrix, [:CompileReverseDiff, :NumClusters, :TrainingSize, :OutputActivationFunction, :NoiseX, :NoiseY, :BalancedAccuracy, :F1Score, :Elapsed])
    CSV.write("./Experiments/stats.csv", df)
end

df = CSV.read("./Experiments/stats.csv", DataFrame, header=1)
names(df)

groupby(df, [:CompileReverseDiff, :TrainingSize])
println(combine(groupby(df, [:CompileReverseDiff, :TrainingSize, :OutputActivationFunction]), :BalancedAccuracy => mean, :BalancedAccuracy => std, "WeightedF1" => mean, "WeightedF1" => std, :Elapsed => mean, :Elapsed => std))
println(combine(groupby(df, [:CompileReverseDiff, :NoiseX, :NoiseY, :TrainingSize, :OutputActivationFunction]), :BalancedAccuracy => mean, :BalancedAccuracy => std, "WeightedF1" => mean, "WeightedF1" => std, :Elapsed => mean, :Elapsed => std))

filtered_df = filter([:CompileReverseDiff, :NoiseX] => (x, y) -> x == false && y == true, df)
println(combine(groupby(filtered_df, [:TrainingSize, :OutputActivationFunction]), :BalancedAccuracy => mean, :BalancedAccuracy => std, "WeightedF1" => mean, "WeightedF1" => std, :Elapsed => mean, :Elapsed => std))

Best = "Compile_ReverseDiff = false, NoiseX =true, NoiseY =false, OutputActivationFunction=Softmax and we conclude that the Aleatoric Uncertainty is just the inverse of Predicted Probability, and the Epistemic Uncertainty is proportional to Std Predicted Probability, More data samples will reduce the epistemic uncertainty in that space."