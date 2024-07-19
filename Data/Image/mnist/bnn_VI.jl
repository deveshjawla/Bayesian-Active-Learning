using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(2; exeflags=`--project`)

PATH = @__DIR__
cd(PATH)
@everywhere begin
	using DistributionsAD
	using Turing
	using Flux

	using ReverseDiff
	# Use reverse_diff due to the number of parameters in neural networks.
	Turing.setadbackend(:reversediff)
	# using LazyArrays
end
using DataFrames
using CSV
using DelimitedFiles
using Statistics, Random


include("../../BNNUtils.jl")
include("../../BNN_Query.jl")
include("../../DataUtils.jl")
include("../../ScoringFunctions.jl")
include("../../AcquisitionFunctions.jl")

###
### Data
###

n_input = 784

train = CSV.read("mnist_train.csv", DataFrame, header=1, skipto=59000)
train.label .+= 1
train_y = select(train, [:label])
train_x = select(train, Not([:label]))

test = CSV.read("mnist_test.csv", DataFrame, header=1, skipto=9000)
test.label .+= 1
test_y = select(test, [:label])
test_x = select(test, Not([:label]))


train_x = Array(Matrix(train_x)') ./ 255
test_x = Array(Matrix(test_x)') ./ 255
train_y = train_y.label
test_y = test_y.label
train_x = reshape(train_x, n_input,1002)
test_x = reshape(test_x, n_input,1002)
# train_x = reshape(train_x, 28,28,1,1002)
# test_x = reshape(test_x, 28,28,1,1002)
# train_x=mapslices(permutedims, train_x, dims=[1,2])
# test_x=mapslices(permutedims, test_x, dims=[1,2])
train_y= permutedims(train_y)
test_y= permutedims(test_y)

# using Gadfly, Cairo, Fontconfig
# set_default_plot_size(6inch, 4inch)
# spy(train_x[:,:,1,4])
###
### Dense Network specifications
###

input_size = size(train_x)[1]
n_output = lastindex(unique(train_y))

@everywhere begin
input_size = $input_size
n_output = $n_output
# neural_network = Chain(Conv((3,3), 1 =>8, relu), Conv((3,3), 8=>8, relu),  Conv((5,5), 8=>8, relu),Conv((5,5), 8=>8, relu), Conv((7,7), 8=>8, relu), Conv((7,7), 8=>8, relu), Flux.flatten, Dense(128 => 10, relu), Dense(10 => 10), softmax)

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:72], 3,3,1,8)
#     b0 = θ[73:80]
#     W1 = reshape(θ[81:656], 3,3,8,8)
#     b1 = θ[657:664]
#     W2 = reshape(θ[665:2264], 5,5,8,8)
#     b2 = θ[2265:2272]
# 	W3 = reshape(θ[2273:3872], 5,5,8,8)
#     b3 = θ[3873:3880]
# 	W4 = reshape(θ[3881:7016], 7,7,8,8)
#     b4 = θ[7017:7024]
# 	W5 = reshape(θ[7025:10160], 7,7,8,8)
#     b5 = θ[10161:10168]

# 	W6 = reshape(θ[10169:11448], 10, 128)
#     b6 = θ[11449:11458]
# 	W7 = reshape(θ[11459:11558], 10, 10)
#     b7 = θ[11559:11568]
#     model = Chain(
#         Conv(W0, b0, relu),
#         Conv(W1, b1, relu),
#         Conv(W2, b2, relu),
#         Conv(W3, b3, relu),
#         Conv(W4, b4, relu),
# 		Conv(W5, b5, relu),
# 		Flux.flatten,
# 		Dense(W6, b6, relu),
# 		Dense(W7, b7),
# 		softmax
#     )
#     return model
# end

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:25088], 32, 784)
	b0 = θ[25089:25120]
	W1 = reshape(θ[25121:25440], 10, 32)
	b1 = θ[25441:25450]
	W2 = reshape(θ[25451:25550], 10, 10)
	b2 = θ[25551:25560]
	model = Chain(
		Dense(W0, b0, relu),
		Dense(W1, b1, relu),
		Dense(W2, b2),
		softmax
	)
	return model
end

# # Extract weights and a helper function to reconstruct NN from weights
# parameters_initial, reconstruct = Flux.destructure(neural_network)

# total_num_params = lastindex(parameters_initial) # number of paraemters in NN
total_num_params = 25560
end

include("../../BayesianModelMultiProc.jl")

train_x, train_y = Float32.(train_x), Int.(train_y)
location_prior, scale_prior = zeros(Float32, total_num_params), ones(Float32, total_num_params)

@everywhere begin
location_prior = $location_prior
scale_prior = $scale_prior
train_x = $train_x
train_y = $train_y
# reconstruct = $reconstruct
model = bayesnnMVG(train_x, train_y, location_prior, scale_prior)
end
nsteps = 1000
chain_timed = @timed sample(model, NUTS(), 10)
###
### Perform Inference using VI
###

# using Turing, AdvancedVI
using Turing: Variational
Turing.setadbackend(:forwarddiff)
# q0 = Variational.meanfield(model)
advi = ADVI(10, 1000)
# opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(model, advi)


elbo(advi, q, m, 1000)

