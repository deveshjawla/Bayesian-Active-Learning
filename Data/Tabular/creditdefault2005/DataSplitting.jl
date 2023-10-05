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
rename!(df, "default payment next month" => :label)

# CSV.write("data.csv", df)



include("../../../DataUtils.jl")
train, test = split_data(df)
train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)

