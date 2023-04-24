using Random
using Turing
using Turing: Variational

Random.seed!(42);

# generate data
x = randn(2000);

@model function model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:lastindex(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# Instantiate model
m = model(x);

# ADVI
advi = ADVI(10, 1000)
q = vi(m, advi);

histogram(rand(q, 1_000)[1, :])

logpdf(q, rand(q))

var(x), mean(x)

(mean(rand(q, 1000); dims=2)...,)