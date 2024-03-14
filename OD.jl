using MLJ
using OutlierDetection

function gendata(n)
    x1 = randn(Float32, 2, n)
    x2 = randn(Float32, 2, n) .+ [2, 2]
    x3 = randn(Float32, 2, n) .+ [-2, 2]
    y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
    y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
    y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
    hcat(x1, x2, x3), hcat(y1, y2, y3)'
end
# Generate data
n = 200
X, y = gendata(n)

train, test = partition(1:size(y)[2], 0.5, shuffle=true, rng=0)

gmm = OutlierDetectionPython.GMMDetector(n_components = 3) #n_components = number of classes or number of modes of the multivariate gaussian
gmm_raw = machine(gmm, X) |> fit!
# transform data to raw outlier scores based on the test data; note that there
# is no `predict` defined for raw detectors
transform(gmm_raw, X[:,test])

# OutlierDetection.jl provides helper functions to normalize the scores,
# for example using min-max scaling based on the training scores
gmm_probas = machine(ProbabilisticDetector(gmm), X) |> fit!
# predict outlier probabilities based on the test data
# gmm_probs = MLJ.predict(gmm_probas, X[:,test])
gmm_probs = MLJ.transform(gmm_probas, X[:,test])[2]

# # OutlierDetection.jl also provides helper functions to turn scores into classes,
# # for example by imposing a threshold based on the training data percentiles
# gmm_classifier = machine(DeterministicDetector(gmm), X) |> fit!
# # predict outlier classes based on the test data
# gmm_preds = MLJ.predict(gmm_classifier, X[:,test])

xs=-7:0.1:7
ys=-7:0.1:7
heatmap(xs, ys, (x,y) -> MLJ.transform(gmm_probas, reshape([x,y],(:,1)))[2][1]) #plots the outlier probabilities
scatter!(X[:,test][1, gmm_probs .> 0.5], X[:,test][2, gmm_probs .> 0.5], color = :cyan)
scatter!(X[:,test][1, gmm_probs .<= 0.5], X[:,test][2, gmm_probs .<= 0.5], color = :green)

# esad = OutlierDetectionNetworks.ESADDetector(encoder = Chain(Dense(2,5), Dense(5,3)), decoder = Chain(Dense(3,5), Dense(5,2)), opt= Flux.AdaBelief()) #Here we need to define endcoder layers and decoder layers
# esad_raw = OutlierDetection.fit(esad, X, CategoricalArray(argmax.(eachrow(y))), verbosity=0) #Here we have to traine it on labels y, where levels(y) should be "anomaly" and "normal"
# train_scores, test_scores = OutlierDetectionNetworks.transform(esad, esad_raw[1], X)