using DataFrames, DelimitedFiles, CSV, StatsBase
PATH = @__DIR__
cd(PATH)
include("../../../../DataUtils.jl")

df=CSV.read("./Concrete_Data.csv", DataFrame, header=1, normalizenames=true; decimal=',', stripwhitespace=true, delim=';')
columnnames = ["Cement_1_kg", "Blast_Furnace_Slag_2_kg", "Fly_Ash_3_kg", "Water_4_kg", "Superplasticizer_5_kg", "Coarse_Aggregate_6_kg", "Fine_Aggregate_7_kg", "Age_days", "label"]
rename!(df, columnnames)
df = dropmissing(df)
println(describe(df))
float_features = mapcols!(col-> StatsBase.standardize(UnitRangeTransform, col), select(df, Not(:Age_days, :label)))
float_features.Age_days = minmaxscaling(df.Age_days, 365, 1)

float_features.label = df.label

df = float_features

train,test=split_data(df)
# mkpath("./Groups")
CSV.write("./train.csv", train)
CSV.write("./test.csv", test)
