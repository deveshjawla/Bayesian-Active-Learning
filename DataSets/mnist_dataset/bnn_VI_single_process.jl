PATH = @__DIR__
cd(PATH)
	using Turing
	using Flux
	using ReverseDiff

	# Use reverse_diff due to the number of parameters in neural networks.
	Turing.setadbackend(:reversediff)
	# using LazyArrays
using DataFrames
using CSV
using DelimitedFiles
using Statistics, Random


include("../../BNNUtils.jl")
include("../../DataUtils.jl")

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
train_x = reshape(train_x, 28,28,1,1002)
test_x = reshape(test_x, 28,28,1,1002)
train_x=mapslices(transpose, train_x, dims=[1,2])
test_x=mapslices(transpose, test_x, dims=[1,2])
train_y= transpose(train_y)
test_y= transpose(test_y)

# using Gadfly, Cairo, Fontconfig
# set_default_plot_size(6inch, 4inch)
# spy(train_x[:,:,1,4])
###
### Dense Network specifications
###

input_size = size(train_x)[1]
n_output = lastindex(unique(train_y))

neural_network = Chain(Conv((7,7), 1 =>4, relu), Conv((7,7), 4=>4, relu), Conv((7,7), 4=>4, relu), Conv((7,7), 4=>4, relu), Flux.flatten, Dense(64 => 16, relu), Dense(16 => 10, relu), softmax)

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(neural_network)

total_num_params = length(parameters_initial) # number of paraemters in NN

include("../../bayesianmodel.jl")

train_x, train_y = Float32.(train_x), Int.(train_y)
location_prior, scale_prior = zeros(Float32, total_num_params), ones(Float32, total_num_params) .* Float32.(1.0)


model = bayesnnMVG(train_x, train_y, location_prior, scale_prior, reconstruct)

nsteps = 1000
chain_timed = @timed sample(model, NUTS(), 1000)

###
### Perform Inference using VI
###

# q0 = Variational.meanfield(model)
advi = ADVI(10, 1000)
# opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(model, advi)


AdvancedVI.elbo(advi, q, m, 1000)

# params_samples = rand(q, 1000)
# params = mean.(eachrow(params_samples))
model = feedforward(params)
ŷ = model(test_x')
predictions = (ŷ .> 0.5)
# count(ŷ .> 0.7)
# count(test_y)

# using Plots

# q_samples = rand(q, 10_000);

# p1 = histogram(q_samples[1, :], alpha = 0.7, label = "q");

# title!(raw"$\θ_1$")

# p2 = histogram(q_samples[2, :], alpha = 0.7, label = "q");

# title!(raw"$\θ_2$")

# plot(p1, p2)
