using MLJ
using OutlierDetection
using OutlierDetectionPython
using OutlierDetectionData: ODDS
using StatisticalMeasures: area_under_curve

using StatsPlots
PATH = @__DIR__
cd(PATH)


# Generate data
n = 200
X, y = gen_3_clusters(n)

scatter(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1", markerstrokewidth=0.1)
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2", markerstrokewidth=0.1)
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3", markerstrokewidth=0.1)

# use 50% of the data for training
train, test = partition(eachindex(y), 0.5, shuffle=true)

oodd = OutlierDetectionPython.GMMDetector(n_components=3) #n_components = number of classes or number of modes of the multivariate gaussian

# oodd_raw = machine(oodd, X) |> fit!
# # transform data to raw outlier scores based on the test data; note that there
# # is no `predict` defined for raw detectors
# OutlierDetection.transform(oodd_raw, X)

# OutlierDetection.jl provides helper functions to normalize the scores,
# for example using min-max scaling based on the training scores
oodd_probas = machine(ProbabilisticDetector(oodd), X) |> fit!

# predict outlier probabilities based on the test data
# oodd_probs = MLJ.predict(oodd_probas, X)
oodd_probs = OutlierDetection.transform(oodd_probas, X)[2] #probability of being an outlier

# # OutlierDetection.jl also provides helper functions to turn scores into classes,
# # for example by imposing a threshold based on the training data percentiles
# oodd_classifier = machine(DeterministicDetector(oodd), X) |> fit!
# # predict outlier classes based on the test data
# oodd_preds = MLJ.predict(oodd_classifier, X)

xs = -7:0.1:7
ys = -7:0.1:7
heatmap(xs, ys, (x, y) -> OutlierDetection.transform(oodd_probas, reshape([x, y], (:, 1)))[2][1], colorbar_title="Uncertainty", xlabel="x", ylabel="y", dpi=300, size=(800, 800)) #plots the outlier probabilities
scatter!(X[1, y[1, :].==1], X[2, y[1, :].==1], color=:red, label="1", markerstrokewidth=0.1)
scatter!(X[1, y[2, :].==1], X[2, y[2, :].==1], color=:green, label="2", markerstrokewidth=0.1)
scatter!(X[1, y[3, :].==1], X[2, y[3, :].==1], color=:blue, label="3", markerstrokewidth=0.1)
scatter!(X[1, oodd_probs.>0.5], X[2, oodd_probs.>0.5], color=:cyan, label="Anomalies", markerstrokewidth=0.1)
savefig("./OD_oodd.pdf")

# scatter!(X[1, oodd_probs .<= 0.5], X[2, oodd_probs .<= 0.5], color = :green)

# esad = OutlierDetectionNetworks.ESADDetector(encoder = Chain(Dense(2,5), Dense(5,3)), decoder = Chain(Dense(3,5), Dense(5,2)), opt= Flux.AdaBelief()) #Here we need to define endcoder layers and decoder layers
# esad_raw = OutlierDetection.fit(esad, X, CategoricalArray(argmax.(eachrow(y))), verbosity=0) #Here we have to traine it on labels y, where levels(y) should be "anomaly" and "normal"
# train_scores, test_scores = OutlierDetectionNetworks.transform(esad, esad_raw[1], X)