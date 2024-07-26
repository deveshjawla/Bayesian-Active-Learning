########################################
# Laplace's Approximation
########################################

#=

References:

# LaplaceRedux.jl Tutorials. "Bayesian MLP Binary Classifier."
https://juliatrustworthyai.github.io/LaplaceRedux.jl/stable/tutorials/mlp/

# Altmeyer, Patrick. Presentation Video. "Effortless Bayesian Deep Learning through Laplace Redux | Patrick Altmeyer | JuliaCon 2022".
https://youtu.be/oWko8FRj_64

# Altmeyer, Patrick. Presentation Slides. "Effortless Bayesian Deep Learning through Laplace Redux." 2022.
https://juliatrustworthyai.github.io/LaplaceRedux.jl/stable/resources/juliacon22/presentation.html#/title-slide

=#

########################################
# Packages
########################################

# load packages

using LaplaceRedux, Flux, Plots

using Flux: params

using Flux.Losses: binarycrossentropy

using Flux.Optimise: Adam, update!

using Statistics

using Random;
Random.seed!(2);

########################################
# Data
########################################

# generate data

xs, ys = LaplaceRedux.Data.toy_data_non_linear(200)

# prep data for Flux.jl

xs = [Float32.(xs[i]) for i in 1:lastindex(xs)]

X = hcat(xs...)

ys = Float32.(ys)

data = zip(xs, ys)

# visualize data

feature1 = [xs[i][1] for i in 1:lastindex(xs)]

feature2 = [xs[i][2] for i in 1:lastindex(xs)]

class = Int32.(ys)

p_data = scatter(feature1, feature2;
    group=class,
    color=[2 3],
    title="Toy Data",
    xlabel="Feature 1",
    ylabel="Feature 2",
    xlims=(-6, 6),
    ylims=(-6, 6),
    aspect_ratio=1,
    size=(460, 400),
    legend=(-0.11, 0.94)
)

vline!(p_data, [0];
    color=:black,
    label=""
)

hline!(p_data, [0];
    color=:black,
    label=""
)

########################################
# Flux.jl
########################################

# define artificial neural network

input = 2

hidden = 10

output = 1

model = Chain(
    Dense(input, hidden, sigmoid),
    Dense(hidden, output, sigmoid)
)

# define loss function

loss(x, y) = binarycrossentropy(model(x), y)

avg_loss(data) = mean(map(d -> loss(d[1], d[2]), data))

# assign variables

learning_rate = Float32(0.001)

opt = Adam(learning_rate)

epochs = 100
show_every = epochs / 10
# train model

for epoch in 1:epochs
    # train model
    for d in data
        gs = gradient(params(model)) do
            l = loss(d...)
        end
        update!(opt, params(model), gs)
    end
    # print report
    train_loss = avg_loss(data)
    if epoch % show_every == 0
        println("Epoch = $epoch : Training Loss = $train_loss")
    end
end

# make predictions

test_nn = [
    [-4, 4],
    [0, 0],
    [3, 2]
]

newX_nn = hcat(test_nn...)

newYS_nn = model(newX_nn)

########################################
# LaplaceRedux.jl
########################################

# find laplace approximation

la = Laplace(model;
    likelihood=:classification,
    subset_of_weights=:all
)

fit!(la, data)

# optimize priors

optimize_prior!(la;
    verbose=true,
    n_steps=500
)

include("./LaplaceReduxPlotting.jl")

p_la = plot(la, X, ys;
    title="Laplace Approximation",
    levels=50,
    aspect_ratio=1,
    size=(460, 400),
    legend=(0.02, 0.98)
)

vline!(p_la, [0];
    color=:black,
    label=""
)

hline!(p_la, [0];
    color=:black,
    label=""
)

# make predictions

test_la = [
    [-4, 4],
    [0, 0],
    [3, 2]
]

newX_la = hcat(test_la...)

predict_la = LaplaceRedux.predict(la, newX_la)
