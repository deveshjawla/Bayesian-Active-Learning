using EvidentialFlux
using Flux #Deep learning
using Plots
using LaplaceRedux

PATH = @__DIR__
cd(PATH)


# Generate data
n = 200
X, y = gen_3_clusters(n)

input_size = size(X)[1]
output_size = size(y)[1]

function train(input_size, output_size)
	# Define model
	m = Chain(
		Dense(input_size => 8, relu),
		Dense(8 => 8, relu),
		# Dense(8 => output_size),
		DIR(8 => output_size)
	)

	_, re = Flux.destructure(m)

	opt = Flux.Optimise.AdaBelief()
	p = Flux.params(m)

	least_loss = Inf32
	last_improvement = 0
	optim_params = 0

	# Train it
	epochs = 500
	trnlosses = zeros(epochs)
	for e in 1:epochs
		local trnloss = 0
		grads = Flux.gradient(p) do
			α = m(X)
			# trnloss = Flux.logitcrossentropy(α, y)
			trnloss = dirloss(y, α, e)
			trnloss
		end
		trnlosses[e] = trnloss
		Flux.Optimise.update!(opt, p, grads)

		if trnloss < least_loss
			# @info(" -> New minimum loss! Saving model weights")
			optim_params, _ = Flux.destructure(m)
			least_loss = trnloss
			last_improvement = e
		end

		# If we haven't seen improvement in 5 epochs, drop our learning rate:
		if e - last_improvement >= 50 && opt.eta > 1e-5
			new_eta = opt.eta / 10.0
			@warn(" -> Haven't improved in a while, dropping learning rate to $(new_eta)!")
			opt = Flux.Optimise.AdaBelief(new_eta)
			# After dropping learning rate, give it a few epochs to improve
			last_improvement = e
		end

		if e - last_improvement >= 100
			@warn(" -> We're calling this converged.")
			break
		end
	end
	return re, optim_params
end	

re, optim_params = train(input_size, output_size)
m = re(optim_params)

#Laplace Approximation
la = Laplace(m; likelihood=:classification, subset_of_weights=:last_layer)
fit!(la, zip(collect(eachcol(X)), y))
optimize_prior!(la; verbose=true, n_steps=100)

# _labels = sort(unique(argmax.(eachcol(y))))
# plt_list = []
# for target in _labels
#     plt = plot(la, X, argmax.(eachcol(y)); target=target, clim=(0,1))
#     push!(plt_list, plt)
# end
# plot(plt_list...)

# _labels = sort(unique(argmax.(eachcol(y))))
# plt_list = []
# for target in _labels
#     plt = plot(la, X, argmax.(eachcol(y)); target=target, clim=(0,1), link_approx=:plugin, markersize = 2)
#     push!(plt_list, plt)
# end
# plot(plt_list...)
# savefig("./Relu_LA.pdf")


# predict_la = LaplaceRedux.predict(la, X, link_approx=:probit)
# mapslices(argmax, predict_la, dims=1)
# mapslices(x->1-maximum(x), predict_la, dims=1)
# entropies = mapslices(x -> normalized_entropy(x, output_size), predict_la, dims=1)


xs = -7.0f0:0.10f0:7.0f0
ys = -7.0f0:0.10f0:7.0f0
# heatmap(xs, ys, (x,y) -> 1-maximum(LaplaceRedux.predict(la, vcat(x,y), link_approx=:probit)))
heatmap(xs, ys, (x, y) -> normalized_entropy(LaplaceRedux.predict(la, vcat(x, y), link_approx=:probit), output_size))
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1")
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2")
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3")
savefig("./LA_relu_entropy.pdf")
