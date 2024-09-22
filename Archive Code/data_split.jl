using MLJ: partition
using CSV, DataFrames

PATH = @__DIR__
cd(PATH)

@elapsed train = CSV.read("./secom_data_preprocessed_moldovan2017.csv", DataFrame, header=false)
@elapsed labels = CSV.read("./secom_labels.txt", DataFrame, header=["target", "datetime"])

df = hcat(train, select(labels, :target))

train, test = partition(df, 0.8, shuffle=true, rng=1334)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)