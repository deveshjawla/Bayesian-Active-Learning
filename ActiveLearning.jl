# 1. Data Split
# 	a. Diversity sampling to obtain the initial dataset
# 2. Define functional Model
# 	a. Define the bayesian Model
# 3. LOOP
# 	a. Set prior
# 	b. condition model on acquired samples 
# 		1. Log performance of newly trained model(include batch size)
# 	c. Score the pool
# 	d. Acquire new samples and add to train set & delete from pool
# 	e. Make poterior as the new prior
# 	f. If new data points are to be added to the pool
# 		1. add the new data to the left over pool
# 		2. Re-normalize the whole train+pool dataset, and then apply the same normalization params to the test set


using Distributed
using Turing
num_chains = 8
num_mcsteps = 100
# Add four processes to use for sampling.
addprocs(num_chains; exeflags=`--project`)

using DataFrames
using CSV
using DelimitedFiles
using Random
using StatsBase
using Distances

include("./BNNUtils.jl")
include("./ALUtils.jl")
include("./Calibration.jl")
include("./DataUtils.jl")
include("./ScoringFunctions.jl")
include("./AcquisitionFunctions.jl")

experiments = "experiments"

PATH = @__DIR__
cd(PATH)

###
### Data
###

include()

###
### Dense Network specifications(Functional Model)
###

input_size = size(pool[1])[1]
n_output = lastindex(unique(pool[2]))
# println("The number of input features are $input_size")
# println("The number of outputs are $n_output")


@everywhere begin
    input_size = $input_size
    n_output = $n_output
    l1, l2, l3, l4 = 5, 5, 5, 5
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    nl3 = l2 * l3 + l3
    nl4 = l3 * l4 + l4
    n_output_layer = l4 * n_output + n_output

    total_num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

    using Flux, Turing

    include()

    # setprogress!(false)
    # using Zygote
    # Turing.setadbackend(:zygote)
    using ReverseDiff
    Turing.setadbackend(:reversediff)

    #Here we define the layer by layer initialisation
    sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + n_output)) * ones(n_output_layer))
    # sigma = ones(total_num_params)

end

include("../../bayesianmodel.jl")

begin
	kpi_df = Array{Any}(missing, 0, 6)
    for acquisition_size in [10]
        for acq_func in ["PowerBALD", "Random"]
            name_exp = "$(acq_func)_$(acquisition_size)_with_$(num_mcsteps)_MCsteps"
            mkpath("./experiments/$(name_exp)/predictions")
            mkpath("./experiments/$(name_exp)/classification_performance")
            mkpath("./experiments/$(name_exp)/convergence_statistics")
            mkpath("./experiments/$(name_exp)/independent_param_matrix_all_chains")
            mkpath("./experiments/$(name_exp)/log_distribution_changes")
            mkpath("./experiments/$(name_exp)/query_batch_class_distributions")


            AL_iteration = 1
            prior = 0
            new_pool, new_prior, param_matrix, new_training_data = 0, 0, 0, 0
            for AL_iteration = 1:round(Int, total_pool_samples / acquisition_size, RoundUp)
                if AL_iteration == 1
                    μ_prior, σ_prior = zeros(total_num_params), sigma
                    # @everywhere μ_prior = $μ_prior
                    # @everywhere σ_prior = $σ_prior
                    prior = (μ_prior, σ_prior)
                    # @everywhere prior = $prior
                    new_pool, new_prior, param_matrix, new_training_data = diversity_sampling(prior, pool, input_size, n_output, AL_iteration, test, name_exp, acquisition_size=acquisition_size, nsteps=num_mcsteps, n_chains=num_chains)
				elseif size(new_pool)[2] > acquisition_size
                    new_prior = (new_prior[1], sigma)
                    new_pool, new_prior, param_matrix, new_training_data = uncertainty_sampling(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=acquisition_size, nsteps=num_mcsteps, n_chains=num_chains)
                elseif size(new_pool)[2] <= acquisition_size && size(new_pool)[2] > 0
                    new_prior = (new_prior[1], sigma)
                    new_pool, new_prior, param_matrix, new_training_data = uncertainty_sampling(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=size(new_pool)[2], nsteps=num_mcsteps, n_chains=num_chains)
                    println("Pool exhausted")
                end
                AL_iteration += 1
            end

            n_acq_steps = round(Int, total_pool_samples / acquisition_size, RoundUp)

            for al_step = 1:n_acq_steps
                data = Array{Any}(undef, 5, num_chains)
                for i = 1:num_chains
                    m = readdlm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
                    data[:, i] = m[:, 2]
                    rm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv")
                end
                d = mean(data, dims=2)
                writedlm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", d)
            end

            performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
            for al_step = 1:n_acq_steps
                m = readdlm("./experiments/$(name_exp)/classification_performance/$(al_step).csv", ',')
                performance_data[1, al_step] = m[2, 2]#AcquisitionSize
                cd = readdlm("./experiments/$(name_exp)/query_batch_class_distributions/$(al_step).csv", ',')
                performance_data[2, al_step] = cd[1, 2]#ClassDistEntropy
                performance_data[3, al_step] = m[3, 2] #Accuracy
                c = readdlm("./experiments/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", ',')
                performance_data[4, al_step] = c[1] #Elapsed
            end
            writedlm("./experiments/$(name_exp)/kpi.csv", performance_data, ',')

			# kpi = readdlm("./experiments/$(name_exp)/kpi.csv", ',')
			kpi = copy(performance_data)
			temporary_vector = Array{Int}(undef, n_acq_steps)
			for i = 1:n_acq_steps
				temporary_vector[i] = sum(kpi[1, 1:i])
			end
			# println(collect(enumerate(temporary_vector)))
			for (i, j) in enumerate(temporary_vector)
				kpi_df = vcat(kpi_df, [acq_func acquisition_size j kpi[2, i] kpi[3, i] kpi[4, i]])
			end
        end
    end
	df = DataFrame(kpi_df, [:AcquisitionFunction, :AcquisitionSize, :CumTrainedSize, :ClassDistEntropy, :Accuracy, :Elapsed])
    CSV.write("./experiments/df.csv", df)
end

# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chain[:lp])
# i = i.I[1]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chain, :θ).value
# θ[i, :]


PATH = @__DIR__
cd(PATH)
using Gadfly, Cairo, Fontconfig, DataFrames, CSV
set_default_plot_size(6inch, 12inch)
df = CSV.read("./experiments/df.csv", DataFrame, header=1)
for (j, i) in enumerate(groupby(df, :AcquisitionSize))
    fig1a = plot(i, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]))
    fig1b = plot(i, x=:CumTrainedSize, y=:ClassDistEntropy, color=:AcquisitionFunction, Geom.step)
    fig1c = plot(i, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.step, Guide.ylabel("Time Elapsed Training (seconds)"))
    vstack(fig1a, fig1b, fig1c) |> PNG("./experiments/$(j).png")
end