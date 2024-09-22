using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(5; exeflags=`--project`)

PATH = @__DIR__
cd(PATH)

include("../../BNNUtils.jl")
include("../../ALUtils.jl")
include("../../Calibration.jl")
include("../../DataUtils.jl")
include("../../ScoringFunctions.jl")
include("../../AcquisitionFunctions.jl")

###
### Data
###
using DataFrames
using CSV
using DelimitedFiles

shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)

n_input = 10


pool = CSV.read("train.csv", DataFrame, header=1)
pool[pool.stroke.==0, :stroke] .= 2
pool = select(pool, vcat(shap_importances.feature_name[1:n_input], "stroke"))
pool = data_balancing(pool, balancing="undersampling", positive_class_label=1, negative_class_label=2)

test = CSV.read("test.csv", DataFrame, header=1)
test[test.stroke.==0, :stroke] .= 2
test = select(test, vcat(shap_importances.feature_name[1:n_input], "stroke"))
test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)


pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]

###
### Dense Network specifications(Functional Model)
###
input_size = size(pool[1])[1]
n_output = lastindex(unique(pool[2]))
println("The number of input features are $input_size")
println("The number of outputs are $n_output")
@everywhere begin
    input_size = $input_size
    n_output = $n_output
    l1, l2, l3, l4 = 20, 20, 20, 20
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    nl3 = l2 * l3 + l3
    nl4 = l3 * l4 + l4
    n_output_layer = l4 * n_output + n_output

    total_num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

    using Flux

    function feedforward(θ::AbstractVector)
        W0 = reshape(θ[1:200], 20, 10)
        b0 = θ[201:220]
        W1 = reshape(θ[221:620], 20, 20)
        b1 = θ[621:640]
        W2 = reshape(θ[641:1040], 20, 20)
        b2 = θ[1041:1060]
        W3 = reshape(θ[1061:1460], 20, 20)
        b3 = θ[1461:1480]
        W4 = reshape(θ[1481:1520], 2, 20)
        b4 = θ[1521:1522]
        model = Chain(
            Dense(W0, b0, relu),
            Dense(W1, b1, relu),
            Dense(W2, b2, relu),
            Dense(W3, b3, relu),
            Dense(W4, b4),
            softmax
        )
        return model
    end

    # nn_initial = Chain(Dense(input_size, l1, relu), Dense(l1, l2, relu), Dense(l2, l3, relu), Dense(l3, l4, relu), Dense(l4, n_output, relu), softmax)

    # # Extract weights and a helper function to reconstruct NN from weights
    # parameters_initial, reconstruct = Flux.destructure(nn_initial)

    # total_num_params = length(parameters_initial) # number of paraemters in NN



    ###
    ### Bayesian Network specifications
    ###
    using Turing

    # setprogress!(false)
    # using Zygote
    # Turing.setadbackend(:zygote)
    using ReverseDiff
    Turing.setadbackend(:reversediff)

    # sigma = 0.2

    #Here we define the layer by layer initialisation
    sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
end

include("../../bayesianmodel.jl")

let
    for acquisition_size in [10, 20, 30, 40, 50, total_pool_samples]
        name_exp = "active_learning_$(acquisition_size)"
        mkpath("./experiments/$(name_exp)/predictions")
        mkpath("./experiments/$(name_exp)/classification_performance")
        mkpath("./experiments/$(name_exp)/convergence_statistics")
        mkpath("./experiments/$(name_exp)/independent_param_matrix_all_chains")
        AL_iteration = 1
        new_pool, new_prior, param_matrix = 0, 0, 0
		for AL_iteration = 1:round(Int, total_pool_samples/acquisition_size, RoundUp)
            if AL_iteration == 1
                μ_prior, σ_prior = zeros(total_num_params), ones(total_num_params) .* sigma
                # @everywhere μ_prior = $μ_prior
                # @everywhere σ_prior = $σ_prior
                prior = (μ_prior, σ_prior)
                # @everywhere prior = $prior
                new_pool, new_prior, param_matrix = diversity_sampling(prior, pool, input_size, n_output, AL_iteration, test, name_exp, acquisition_size=acquisition_size)
            elseif size(new_pool)[2] < acquisition_size && size(new_pool)[2] != 0
				new_pool, new_prior, param_matrix = uncertainty_sampling(new_prior, new_pool, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=size(new_pool)[2])
				println("Pool exhausted")
			elseif size(new_pool)[2] > acquisition_size
                new_pool, new_prior, param_matrix = uncertainty_sampling(new_prior, new_pool, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=acquisition_size)
            end
            AL_iteration += 1
        end
    end
end

# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chain[:lp])
# i = i.I[1]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chain, :θ).value
# θ[i, :]

# PATH = @__DIR__
# cd(PATH)
# using DelimitedFiles
# begin
# 	data = Array{Any}(undef, 5, 5)
#     for i = 1:5
#         m = readdlm("./experiments/$(name_exp)/statistics_chain_$(i).csv", ',')
#         data[:, i] = m[:, 2]
#     end
#     d = mean(data, dims=2)
#     writedlm("./experiments/$(name_exp)/statistics_chain_mean.txt", d)
# end