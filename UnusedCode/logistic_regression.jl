
# Retrieve the number of observations.
l, n = size(train_x)

σ = sqrt(2 / n)

# Bayesian logistic regression (LR)
@model function logistic_regression(x, y, n, σ)
    intercept ~ Normal(0, σ)

    θ ~ MvNormal(zeros(454), σ .* ones(454))

    for i in 1:n
        v = logistic(intercept + sum(θ .* x[i, :]))
        y[i] ~ Bernoulli(v)
    end
end

# Sample using HMC.
m = logistic_regression(train_x, train_y, l, σ)
chain_timed = @timed sample(m, NUTS(50, 0.65), 100)
chain = chain_timed.value
elapsed = chain_timed.time

# describe(chain)
intercept = chain[:intercept]
θ = MCMCChains.group(chain, :θ).value
intercepts = collect.(eachrow(intercept[:, 1]))
thetas = collect.(eachrow(θ[:, :, 1]))

if sum(map(x->count(isnan.(x)), intercepts)) > 1 || sum(map(x->count(isnan.(x)), thetas)) > 1
    error("NaNs found in Params")
end

using MLJ

experiment = "lr_3"
mkdir("./test_results_$(experiment)")

function pred_analyzer(test_xs, intercepts, thetas, threshold)
    l = length(intercepts)
    # Retrieve the number of rows.
    n, _ = size(test_xs)

    predictions = Array{Float32,2}(undef, n, l)
    for j in 1:l

        # Generate a vector to store our predictions.
        v = Vector{Int}(undef, n)

        # Calculate the logistic function for each element in the test set.
        for i in 1:n
            num = logistic(intercepts[j][1] + sum(thetas[j] .* test_xs[i, :]))
            if num >= threshold
                v[i] = 1
            else
                v[i] = 0
            end
        end
        predictions[:, j] = v
    end

    predictions_mean = mean(predictions, dims=2)
    predictions_std = std(predictions, dims=2)

    # for each samples mean, std in zip(means, stds)
    # plot(histogram, mean, std)
    # savefig(./plots of each sample)
    # end
    return predictions_mean, predictions_std
end

# Set the prediction threshold.
thresholds = collect(range(0.01, 0.99, step=0.01))

for threshold in thresholds
    
    predictions_mean, predcitions_std = pred_analyzer(test_x, intercepts, thetas, threshold)

	predictions_mean = vec(round.(Int, predictions_mean))

    writedlm("./test_results_$(experiment)/results_$(threshold).txt", [elapsed, f1, mcc, acc, fpr, fnr, tpr, tnr, prec, recall], ',')
end