using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)

using DataFrames
using CSV
using DelimitedFiles
using XGBoost
using Random
using StatsBase
using Flux: softmax

PATH = @__DIR__
cd(PATH)

experiments = "experiments"

include("../../../ALUtils.jl")
include("../../../DataUtils.jl")
include("../../../ScoringFunctions.jl")
include("../../../AcquisitionFunctions.jl")
###
### Data
###

shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
n_input = 30

pool = CSV.read("train.csv", DataFrame, header=1)
pool[pool.target.==-1, :target] .= 2
# pool = select(pool, vcat(shap_importances.feature_name[1:n_input], "target"))
pool = data_balancing(pool, balancing="undersampling", positive_class_label=1, negative_class_label=2)

test = CSV.read("test.csv", DataFrame, header=1)
test[test.target.==-1, :target] .= 2
# test = select(test, vcat(shap_importances.feature_name[1:n_input], "target"))
test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

pool, test_set = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]

input_size = size(pool[1])[1]
n_output = lastindex(unique(pool[2]))

# X=copy(transpose(pool[1]))
# test_x=copy(transpose(test_set[1]))
# Y=vec(Int.(copy(transpose(pool[2])))).-1
# test_y=vec(Int.(copy(transpose(test_set[2])))).-1

pool_x = pool[1]
test_set_x = test_set[1]
pool_y = Int.(pool[2]) .- 1
test_set_y = Int.(test_set[2]) .- 1

pool = (pool_x, pool_y)
test_set = (test_set_x, test_set_y)

let
    for acquisition_size in [10]
        name_exp = "maxentropy_$(acquisition_size)"
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
        new_pool, new_xgb, new_training_data = 0, 0, 0
        new_pool_random, new_xgb_random, new_training_data_random = 0, 0, 0
        for AL_iteration = 1:round(Int, total_pool_samples / acquisition_size, RoundUp)
            if AL_iteration == 1
                new_pool, new_xgb, new_xgb_random, new_training_data = diversity_sampling(pool, input_size, n_output, AL_iteration, test_set, name_exp, acquisition_size=acquisition_size, nsteps=100)
                new_pool_random, new_training_data_random = new_pool, new_training_data
            elseif size(new_pool)[2] <= acquisition_size && size(new_pool)[2] > 0
                new_pool, new_xgb, new_training_data = uncertainty_sampling(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test_set, name_exp, acquisition_size=size(new_pool)[2], nsteps=100)

                new_pool_random, new_xgb_random, new_training_data_random = random_sampling(new_xgb_random, new_pool_random, new_training_data_random, input_size, n_output, AL_iteration, test_set, name_exp_random, acquisition_size=size(new_pool_random)[2], nsteps=100)

                println("Pool exhausted")
            elseif size(new_pool)[2] > acquisition_size
                new_pool, new_xgb, new_training_data = uncertainty_sampling(new_xgb, new_pool, new_training_data, input_size, n_output, AL_iteration, test_set, name_exp, acquisition_size=acquisition_size, nsteps=100)

                new_pool_random, new_xgb_random, new_training_data_random = random_sampling(new_xgb_random, new_pool_random, new_training_data_random, input_size, n_output, AL_iteration, test_set, name_exp_random, acquisition_size=acquisition_size, nsteps=100)
            end
            AL_iteration += 1
        end
    end
end


let
	for acquisition_size in [10]
		name_exps = ["maxentropy_$(acquisition_size)", "random_sampling_$(acquisition_size)"]
		for name_exp in name_exps
            n_acq_steps = round(Int, total_pool_samples / acquisition_size, RoundUp)
            performance_data = Array{Any}(undef, 4, n_acq_steps) #dims=(features, samples(i))
            for al_step = 1:n_acq_steps
                m = readdlm("./$(experiments)/$(name_exp)/classification_performance/$(al_step).csv", ',')
                performance_data[1, al_step] = m[1, 2]
                cl_dist_string = m[2, 2] * "," * m[2, 3]
                class_dict = eval(Meta.parse(cl_dist_string))
                class_dist_ent = normalized_entropy(softmax(collect(values(class_dict))), n_output)
                performance_data[2, al_step] = class_dist_ent
                performance_data[3, al_step] = m[3, 2]
                # c = readdlm("./$(experiments)/$(name_exp)/convergence_statistics/$(al_step)_chain.csv", ',')
                performance_data[4, al_step] = 0
            end
            writedlm("./$(experiments)/$(name_exp)/kpi.csv", performance_data)
        end
    end
end

let
	length_x_axis = round(Int, total_pool_samples/10, RoundUp)
	kpi_matrix = Array{Any}(missing, 7, length_x_axis, 6)
	kpi_df = Array{Any}(missing, 0, 6)
	for (z,acquisition_size) in enumerate([10])
		name_exp = "maxentropy_$(acquisition_size)"
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

using Gadfly, Cairo, Fontconfig, DataFrames, CSV
set_default_plot_size(6inch, 8inch)
df = CSV.read("./$(experiments)/df.csv", DataFrame, header = 1)
for (j,i) in enumerate(groupby(df, :AcquisitionSize))
	fig1a = plot(i, x=:CumTrainedSize, y=:Accuracy, color=:AcquisitionFunction, Geom.point, Geom.line, yintercept=[0.5], Geom.hline(color=["red"], size=[1mm]))
	fig1b = plot(i, x=:CumTrainedSize, y=:ClassDistEntropy, color=:AcquisitionFunction, Geom.step)
	# fig1c = plot(i, x=:CumTrainedSize, y=:Elapsed, color=:AcquisitionFunction, Geom.step,  Guide.ylabel("Time Elapsed Training (seconds)"))
	vstack(fig1a, fig1b) |> PNG("./$(experiments)/$(j).png")
end