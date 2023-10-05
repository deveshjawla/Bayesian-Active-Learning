using DataFrames, DelimitedFiles, CSV, StatsBase

PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")

df = CSV.read("Iris_cleaned.csv", DataFrame, header=1)
rename!(df, :Species => :label)
println(describe(df))
float_features = mapcols(col-> StatsBase.standardize(UnitRangeTransform, col), select(df, Not(:label)))

float_features.label = df.label

df = float_features
train, test = split_data(df)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)
