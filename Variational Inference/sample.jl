using Flux
using StatsPlots
using DataFrames
PATH = @__DIR__
cd(PATH)
# Generate data
include("./../DataUtils.jl")

# Generate data
n = 20
train_x, train_y = gen_3_clusters(n)
train_y = argmax.(eachcol(train_y))

input_size = size(train_x, 1)
output_size = size(train_y, 1)

l1, l2 = 8, 8
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * output_size

num_params = nl1 + nl2 + n_output_layer

include("./../MCMC Variants Comparison/SoftmaxNetwork3.jl")
prior_std = Float32.(sqrt(2) .* vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + output_size)) * ones(n_output_layer)))

using Turing, Flux, ReverseDiff
include("./../BayesianModel.jl")

model = softmax_bnn_noise_x(train_x, train_y, num_params, prior_std)

using Flux, Turing
using Turing.Variational
q0 = Variational.meanfield(model)
advi = ADVI(10, 10_000)
opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(model, advi, q0; optimizer=opt)
z = rand(q, 10_000)
avg = vec(mean(z; dims=2))

q.dist
q.dist.m
q.dist.σ
cov(q.dist)

elbo(advi, q, model, 1000)
using Bijectors
_, sym2range = bijector(model, Val(true))
histogram(z[1, :])
avg[union(sym2range[:noise_x]...)]

function plot_variational_marginals(z, sym2range)
    ps = []

    for (i, sym) in enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)  # <= array of ranges
        if sum(length.(indices)) > 1
            offset = 1
            for r in indices
                p = density(
                    z[r, :];
                    title="$(sym)[$offset]",
                    titlefontsize=10,
                    label="",
                    ylabel="Density"
                )
                push!(ps, p)
                offset += 1
            end
        else
            p = density(
                z[first(indices), :];
                title="$(sym)",
                titlefontsize=10,
                label="",
                ylabel="Density"
            )
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 2000))
end

plot_variational_marginals(z, sym2range.noise_x)