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
# train, test = split_data(df)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)

Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 20
n_folds = 5
fold_size = div(size(df)[1], n_folds)

mkpath("./FiveFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
	train = df[(fold_size*(i-1))+1:fold_size*i, :]
	test = df[Not((fold_size*(i-1))+1:fold_size*i), :]
	CSV.write("./FiveFolds/train_$(i).csv", train)
	CSV.write("./FiveFolds/test_$(i).csv", test)
	# println((fold_size*(i-1))+1:fold_size*i)
end

