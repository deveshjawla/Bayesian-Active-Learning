using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(8; exeflags=`--project`)

PATH = @__DIR__
cd(PATH)

experiments = "4_cumulative_training_data"

include("../../BNNUtils.jl")
include("../../ALUtils.jl")
include("../../Calibration.jl")
include("../../DataUtils.jl")
include("../../ScoringFunctions.jl")
include("../../AcquisitionFunctions.jl")

###
### Data
###
using DataFrames
using CSV
using DelimitedFiles

shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
n_input = 10

pool = CSV.read("train.csv", DataFrame, header=1)
pool[pool.stroke.==0, :stroke] .= 2
pool = select(pool, vcat(shap_importances.feature_name[1:n_input], "stroke"))
pool = data_balancing(pool, balancing="undersampling", positive_class_label=1, negative_class_label=2)

test = CSV.read("test.csv", DataFrame, header=1)
test[test.stroke.==0, :stroke] .= 2
test = select(test, vcat(shap_importances.feature_name[1:n_input], "stroke"))
test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]


using XGBoost

X=copy(transpose(pool[1]))
test_x=copy(transpose(test[1]))
Y=vec(Int.(copy(transpose(pool[2])))).-1
test_y=vec(Int.(copy(transpose(test[2])))).-1

model = xgboost((X, Y), num_round=10, max_depth=20, num_class = 2)

ŷ = XGBoost.predict(model, test_x)

acc = mean(test_y.==ŷ)