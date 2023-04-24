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
# Add four processes to use for sampling.
addprocs(5; exeflags=`--project`)

PATH = @__DIR__
cd(PATH)

experiments = "4_cumulative_training_data"

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


iris = CSV.read("Iris_cleaned.csv", DataFrame, header=1)

target = "Species"
using Random
iris = iris[shuffle(axes(iris, 1)), :]
pool, test = split_data(iris, at=0.8)
n_input = 4

pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]

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

    using Flux

    function feedforward(θ::AbstractVector)
        W0 = reshape(θ[1:20], 5, 4)
        b0 = θ[21:25]
        W1 = reshape(θ[26:50], 5, 5)
        b1 = θ[51:55]
        W2 = reshape(θ[56:80], 5, 5)
        b2 = θ[81:85]
        W3 = reshape(θ[86:110], 5, 5)
        b3 = θ[111:115]
        W4 = reshape(θ[116:130], 3, 5)
        b4 = θ[131:133]
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
    for acquisition_size in [10]
        name_exp = "active_learning_$(acquisition_size)"
        name_exp_random = "random_sampling_$(acquisition_size)"
        mkpath("./$(experiments)/$(name_exp)/predictions")
        mkpath("./$(experiments)/$(name_exp)/classification_performance")
        mkpath("./$(experiments)/$(name_exp)/convergence_statistics")
        mkpath("./$(experiments)/$(name_exp)/independent_param_matrix_all_chains")
        mkpath("./$(experiments)/$(name_exp)/log_distribution_changes")
		
		mkpath("./$(experiments)/$(name_exp_random)/predictions")
        mkpath("./$(experiments)/$(name_exp_random)/classification_performance")
        mkpath("./$(experiments)/$(name_exp_random)/convergence_statistics")
        mkpath("./$(experiments)/$(name_exp_random)/independent_param_matrix_all_chains")
        mkpath("./$(experiments)/$(name_exp_random)/log_distribution_changes")
        AL_iteration = 1
        new_pool, new_prior, param_matrix, new_training_data = 0, 0, 0, 0
		new_pool_random, new_prior_random, param_matrix_random, new_training_data_random = 0,0,0, 0
        for AL_iteration = 1:round(Int, total_pool_samples / acquisition_size, RoundUp)
            if AL_iteration == 1
                μ_prior, σ_prior = zeros(total_num_params), ones(total_num_params) .* sigma
                # @everywhere μ_prior = $μ_prior
                # @everywhere σ_prior = $σ_prior
                prior = (μ_prior, σ_prior)
                # @everywhere prior = $prior
                new_pool, new_prior, param_matrix, new_training_data = diversity_sampling(prior, pool, input_size, n_output, AL_iteration, test, name_exp, acquisition_size=acquisition_size, nsteps=1000)
				new_pool_random, new_prior_random, param_matrix_random, new_training_data_random = new_pool, new_prior, param_matrix, new_training_data
            elseif size(new_pool)[2] < acquisition_size && size(new_pool)[2] > 0
				new_pool, new_prior, param_matrix, new_training_data = uncertainty_sampling(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=size(new_pool)[2], nsteps=1000)

				new_pool_random, new_prior_random, param_matrix_random, new_training_data_random = random_sampling(new_prior_random, new_pool_random, new_training_data_random, input_size, n_output, param_matrix_random, AL_iteration, test, name_exp_random, acquisition_size=size(new_pool_random)[2], nsteps=1000)

				println("Pool exhausted")
			elseif size(new_pool)[2] > acquisition_size
				new_pool, new_prior, param_matrix, new_training_data = uncertainty_sampling(new_prior, new_pool, new_training_data, input_size, n_output, param_matrix, AL_iteration, test, name_exp, acquisition_size=acquisition_size, nsteps=1000)

				new_pool_random, new_prior_random, param_matrix_random, new_training_data_random = random_sampling(new_prior_random, new_pool_random, new_training_data_random, input_size, n_output, param_matrix_random, AL_iteration, test, name_exp_random, acquisition_size=acquisition_size, nsteps=1000)
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


let
	total_pool_samples = 70
	for acquisition_size in [10]
        name_exp = "active_learning_$(acquisition_size)"
		for al_step = 1:round(Int, total_pool_samples/acquisition_size, RoundUp)
			data = Array{Any}(undef, 5, 5)
			for i = 1:5
				m = readdlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
				data[:, i] = m[:, 2]
				rm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv")
			end
			d = mean(data, dims=2)
			writedlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", d)
		end
	end
end


	total_pool_samples = 70
	for acquisition_size in [10]
        name_exp = "random_sampling_$(acquisition_size)"
		for al_step = 2:round(Int, total_pool_samples/acquisition_size, RoundUp)
			try
			data = Array{Any}(undef, 5, 5)
			for i = 1:5
				m = readdlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv", ',')
				data[:, i] = m[:, 2]
				rm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain_$i.csv")
			end
			d = mean(data, dims=2)
			writedlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", d)
			catch
				println("Catching")
			end
		end
	end


PATH = @__DIR__
cd(PATH)
using DelimitedFiles
let
	total_pool_samples = 70
	for acquisition_size in [10]
        name_exp = "active_learning_$(acquisition_size)"
		n_acq_steps = round(Int, total_pool_samples/acquisition_size, RoundUp)
		performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
		for al_step = 1:n_acq_steps
			m = readdlm("./$(experiments)/$(name_exp)/classification_performance/$(al_step).csv", ',')
			performance_data[1, al_step] = m[1, 2]
			cl_dist_string=m[2,2]*","*m[2,3]*","*m[2,4]
			class_dict=eval(Meta.parse(cl_dist_string))
			class_dist_ent=entropy(softmax(collect(values(class_dict))), n_output)
			performance_data[2, al_step] = class_dist_ent
			performance_data[3, al_step] = m[3, 2]
			c = readdlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", ',')
			performance_data[4, al_step] = c[1]
		end
		writedlm("./$(experiments)/$(name_exp)/kpi.csv", performance_data)
	end
end

PATH = @__DIR__
cd(PATH)
using DelimitedFiles
let
	total_pool_samples = 70
	for acquisition_size in [10]
        name_exp = "random_sampling_$(acquisition_size)"
		n_acq_steps = round(Int, total_pool_samples/acquisition_size, RoundUp)
		performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
		for al_step = 1:n_acq_steps
			if al_step==1
				performance_data[:, al_step] = zeros(4)
			else
				m = readdlm("./$(experiments)/$(name_exp)/classification_performance/$(al_step).csv", ',')
				performance_data[1, al_step] = m[1, 2]
				cl_dist_string=m[2,2]*","*m[2,3]*","*m[2,4]
				class_dict=eval(Meta.parse(cl_dist_string))
				class_dist_ent=entropy(softmax(collect(values(class_dict))), n_output)
				performance_data[2, al_step] = class_dist_ent
				performance_data[3, al_step] = m[3, 2]
				c = readdlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", ',')
				performance_data[4, al_step] = c[1]
			end
		end
		writedlm("./$(experiments)/$(name_exp)/kpi.csv", performance_data)
	end
end

using DataFrames, CSV
let
	total_pool_samples = 70
	length_x_axis = round(Int, total_pool_samples/10, RoundUp)
	kpi_matrix = Array{Any}(missing, 7, length_x_axis, 6)
	kpi_df = Array{Any}(missing, 0, 6)
	for (z,acquisition_size) in enumerate([10])
		name_exp = "active_learning_$(acquisition_size)"
		name_exp_random = "random_sampling_$(acquisition_size)"
		n_acq_steps = round(Int, total_pool_samples/acquisition_size, RoundUp)
		kpi = readdlm("./$(experiments)/$(name_exp)/kpi.csv", '\t')
		kpi_random = readdlm("./$(experiments)/$(name_exp_random)/kpi.csv", '\t')
		temporary_vector = Array{Int}(undef, n_acq_steps)
		for i=1:n_acq_steps
			temporary_vector[i] = sum(kpi[1,1:i])
		end
		# println(collect(enumerate(temporary_vector)))
		for (i,j) in enumerate(temporary_vector)
			kpi_matrix[1,i,z] = j
			# println("kpi",kpi_matrix)
			kpi_matrix[2:4,i,z] = kpi[2:4, i]
			if i == 1
				kpi_matrix[5:7,i,z] = kpi[2:4, 1]
				kpi_df = vcat(kpi_df, ["PowerBALD" acquisition_size j kpi[2, i] kpi[3, i] kpi[4, i]])
			else
				kpi_matrix[5:7,i,z] = kpi_random[2:4, i]
				kpi_df = vcat(kpi_df, ["PowerBALD" acquisition_size j kpi[2, i] kpi[3, i] kpi[4, i]])
				kpi_df = vcat(kpi_df, ["Random" acquisition_size j kpi_random[2, i] kpi_random[3, i] kpi_random[4, i]])
			end
		end
	end
	df=DataFrame(kpi_df, [:AcquisitionFunction,:AcquisitionSize, :CumTrainedSize, :ClassDistEntropy, :Accuracy, :Elapsed])
	CSV.write("./$(experiments)/df.csv", df)
end

using Gadfly, Cairo, Fontconfig
set_default_plot_size(6inch, 4inch)
df = CSV.read("./$(experiments)/df.csv", DataFrame, header = 1)
for (j,i) in enumerate(groupby(df, :AcquisitionSize))
	plot(i, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm])) |> PNG("./$(experiments)/$(j).png")
end