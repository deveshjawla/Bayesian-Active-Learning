using DataFrames, DelimitedFiles, CSV

PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")

df = CSV.read("./data_banknote_authentication.txt", DataFrame, header=false)
rename!(df, :Column5 => :label)
df[df.label.==0, :label] .= 2
train, test = split_data(df)
train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)

