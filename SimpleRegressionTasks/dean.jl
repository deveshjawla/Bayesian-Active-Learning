using Flux
using Plots
using Distributions

using Turing

f(x) = cos(x) + rand(Normal(0, 0.1))

xTrain = collect(-3:0.1:3)
yTrain = f.(xTrain)
plot(xTrain, yTrain, seriestype=:scatter, label="Train Data")
plot!(xTrain, cos.(xTrain), label="Truth")

x = rand(Normal(), 100)
y = f.(x)
train_data = Iterators.repeated((Array(x'), Array(y')), 100);

function unpack(nn_params::AbstractVector)
    W₁ = reshape(nn_params[1:2], 2, 1);   
    b₁ = reshape(nn_params[3:4], 2)
    
    W₂ = reshape(nn_params[4:5], 1, 2); 
    b₂ = [nn_params[6]]
    
    return W₁, b₁, W₂, b₂
end

function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, W₂, b₂ = unpack(nn_params)
    nn = Chain(Dense(W₁, b₁, tanh), Dense(W₂, b₂))
    return nn(xs)
end

alpha = 0.1
sig = sqrt(1.0 / alpha)

@model bayes_nn(xs, ys) = begin
    
    nn_params ~ MvNormal(zeros(6), sig .* ones(6)) #Prior
    
    preds = nn_forward(xs, nn_params) #Build the net
    sigma ~ Gamma(0.01, 1/0.01) # Prior for the variance
    for i = 1:length(ys)
        ys[i] ~ Normal(preds[i], sigma)
    end
end;

N = 5000
ch1 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);
ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N);

lp, maxInd = findmax(ch1[:lp])

params, internals = ch1.name_map
bestParams = map(x-> ch1[x].data[maxInd], params[1:6])
plot(x, cos.(x), seriestype=:line, label="True")
plot!(x, Array(nn_forward(hcat(x...), bestParams)'), 
      seriestype=:scatter, label="MAP Estimate")

	  xPlot = sort(x)

sp = plot()

for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    paramSample = map(x-> ch1[x].data[i], params)
    plot!(sp, xPlot, Array(nn_forward(hcat(xPlot...), paramSample)'), 
        label=:none, colour="blue")
    
end

plot!(sp, x, y, seriestype=:scatter, label="Training Data", colour="red")

lPlot = plot(ch1[:lp], label="Chain 1", title="Log Posterior")
plot!(lPlot, ch2[:lp], label="Chain 2")

sigPlot = plot(ch1[:sigma], label= "Chain 1", title="Variance")
plot!(sigPlot, ch2[:sigma], label="Chain 2")

plot(lPlot, sigPlot)