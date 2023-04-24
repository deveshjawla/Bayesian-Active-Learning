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


iris = CSV.read("Iris_cleaned.csv", DataFrame, header=1)

target = "Species"
using Random
iris = iris[shuffle(axes(iris, 1)), :]
pool, test = split_data(iris, at=0.8)
n_input = 4

pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]

using XGBoost

X=copy(transpose(pool[1]))
test_x=copy(transpose(test[1]))
Y=vec(Int.(copy(transpose(pool[2])))).-1
test_y=vec(Int.(copy(transpose(test[2])))).-1

model = xgboost((X, Y), num_round=10, max_depth=3, objective="multi:softmax", num_class = 3)

ŷ = XGBoost.predict(model, test_x)

acc = mean(test_y.==ŷ)