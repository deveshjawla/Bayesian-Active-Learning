using DataFrames, DelimitedFiles, CSV
using StatsBase
PATH = @__DIR__
cd(PATH)

df = CSV.read("data.csv", DataFrame, header=1)

println(describe(df))
mapcols(col -> println(unique(col)), select(df, [:SEX, :EDUCATION, :MARRIAGE]))
# select!(df, Not(:ID))
# df[df.label.==2, :label] .= 0
train_groups = groupby(df, [:SEX, :MARRIAGE])
ttrain = transform(train_groups, :label=>mean)
train = select(ttrain, Not([:SEX, :MARRIAGE]))
rename!(train, :label_mean => :group_weight)
train = train[:, [1:21..., 23, 22]]
a = mapcols(col-> StatsBase.standardize(UnitRangeTransform, Float64.(col)), select(train, 2,3))
b = mapcols(col-> StatsBase.standardize(ZScoreTransform, Float64.(col)), select(train, 4:15))
c = mapcols(col-> StatsBase.standardize(ZScoreTransform, Float64.(col); center=false), select(train, 1, 16:21))

float_features = hcat(a, b, c)
float_features.group_weight = train.group_weight
float_features.label = train.label

df = float_features


# train_keys=keys(train_groups)
combines_train = combine(train_groups, :label=> mean)
combines_train = combine(train_groups, :label=> lastindex)

# CSV.write("data.csv", df)



include("../../../DataUtils.jl")
# train, test = split_data(df)
# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)

Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 80
n_folds = 5
fold_size = div(size(df)[1], n_folds)

mkpath("./FiveFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
	train = df[(fold_size*(i-1))+1:fold_size*i, :]
	train, leftovers = balance_binary_data(train)
	test = df[Not((fold_size*(i-1))+1:fold_size*i), :]
	test = vcat(test, leftovers)
	CSV.write("./FiveFolds/train_$(i).csv", train)
	CSV.write("./FiveFolds/test_$(i).csv", test)
	# println((fold_size*(i-1))+1:fold_size*i)
end
