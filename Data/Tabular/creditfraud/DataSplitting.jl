using DataFrames, DelimitedFiles, CSV
using StatsBase
PATH = @__DIR__
cd(PATH)

df = CSV.read("data.csv", DataFrame)

println(describe(df))
select!(df, Not([:Time, :Amount]))
rename!(df, "Class" => :label)
df[df.label.==0, :label] .= 2

CSV.write("data.csv", df)



train,test=split_data(df)
train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
