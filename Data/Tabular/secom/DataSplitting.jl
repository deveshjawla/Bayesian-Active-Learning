using DataFrames, DelimitedFiles, CSV, StatsBase
PATH = @__DIR__
cd(PATH)

# using MultivariateStats

# M = fit(PCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim=17)
# train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))

# CSV.write("../secomPCA/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
# CSV.write("../secomPCA/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))


df = CSV.read("secom_data_preprocessed_moldovan2017.csv", DataFrame, header=false)
labels = readdlm("secom_labels.txt", ' ')
println(describe(df))

to_be_scaled = findall(x->minimum(x)>=0, eachcol(df))
to_be_normalized = findall(x->minimum(x)<0, eachcol(df))
a = mapcols(col-> StatsBase.standardize(UnitRangeTransform, col), select(df, to_be_scaled))
# b = mapcols(col-> StatsBase.standardize(ZScoreTransform, Float64.(col)), select(df, to_be_normalized))
b = mapcols(col-> StatsBase.standardize(UnitRangeTransform, Float64.(col), unit=false), select(df, to_be_normalized))

float_features = hcat(a, b)

labels = labels[:,1]
labels = labels.==1
float_features.labels = Int.(labels)
float_features.labels[float_features.labels.==0] .= 2

include("../../../DataUtils.jl")
train, test = split_data(float_features)
train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)