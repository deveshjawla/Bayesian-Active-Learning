using Distributed
addprocs(10, exeflags="--project=$(Base.active_project())")
@everywhere begin
    include("../TreeEnsemble.jl")
    using ShapML, Gadfly
    using DataFrames, DelimitedFiles, Statistics, CSV
    using Random
end
@everywhere using .TreeEnsemble


PATH = @__DIR__
cd(PATH)

train = CSV.read("./train.csv", DataFrame)

train[train.stroke.==0, :stroke] .= 2

function data_balancing(data_xy; balancing=true)
    if balancing == true
        normal_data = data_xy[data_xy[:, end].==2.0, :]
        anomaly = data_xy[data_xy[:, end].==1.0, :]
        data_xy = vcat(normal_data[1:size(anomaly)[1], :], anomaly)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    else
        nothing
    end
    data_x = data_xy[:, 1:end-1]
    data_y = select(data_xy, :stroke)
    return data_x, data_y
end

train_x, train_y = data_balancing(train, balancing=true)

## -------------- fit data  -------------- ##
# dtc = DecisionTreeClassifier(random_state=42)

max_features = ceil(sqrt(size(train_x)[2]))
n_trees = 300
min_samples_leaf = 3
rfc = RandomForestClassifier(random_state=42, n_trees=n_trees, bootstrap=true, oob_score=true, max_features=max_features, min_samples_leaf=min_samples_leaf)

classifier = rfc

print("fitting time           ");
elapsed = @elapsed fit!(classifier, train_x, train_y)
println(elapsed)

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
@everywhere function predict_function(model, data)
    data_pred = DataFrame(y_pred=predict(model, data))
    return data_pred
end

# ShapML setup.
explain = copy(train_x) # Compute Shapley feature-level predictions for 300 instances.

reference = copy(train_x)  # An optional reference population to compute the baseline prediction.

sample_size = 100  # Number of Monte Carlo samples.
#------------------------------------------------------------------------------
# Compute stochastic Shapley values.
data_shap = ShapML.shap(explain=explain,
    reference=nothing,
    model=classifier,
    predict_function=predict_function,
    sample_size=sample_size,
    parallel=:samples,# Parallel computation over "sample_size".
    seed=1
)

show(data_shap, allcols=true)

gd = groupby(data_shap, :feature_name)
data_plot = combine(gd, :shap_effect => x -> mean(abs.(x)))

data_plot = sort(data_plot, order(:shap_effect_function, rev=true))

CSV.write("./shap_importances.csv", data_plot)

baseline = round(data_shap.intercept[1], digits=1)

p = plot(data_plot[:, :], y=:feature_name, x=:shap_effect_function, Coord.cartesian(yflip=true),
    Scale.y_discrete, Geom.bar(position=:dodge, orientation=:horizontal),
    Theme(bar_spacing=1mm),
    Guide.xlabel("|Shapley effect| (baseline = $baseline)"), Guide.ylabel(nothing),
    Guide.title("Feature Importance - Mean Absolute Shapley Value"))

import Cairo, Fontconfig
draw(PNG("./imp_shap_features.png", 6inch, 4inch), p)
