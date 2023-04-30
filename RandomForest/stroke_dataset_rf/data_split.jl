using MLJ: partition
using CSV, DataFrames

PATH = @__DIR__
cd(PATH)

df = CSV.read("./stroke_dataset_categorised.csv", DataFrame)
df = select(df, Not(:id))

train, test = partition(df, 0.8, shuffle=true, rng=1334)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)