using DataFrames, DelimitedFiles, CSV
using StatsBase, MLJ, Flux
PATH = @__DIR__
cd(PATH)

df = CSV.read("data.csv", DataFrame, header=1)
include("../../../DataUtils.jl")

println(describe(df))
mapcols(col -> println(unique(col)), select(df, [:EDUCATION, :MARRIAGE]))


binary_features = mapcols(to_binary_vector, select(df, [:SEX]))

# select!(df, Not(:ID))
# df[df.label.==2, :label] .= 0
# train_groups = groupby(df, [:SEX, :MARRIAGE])
# ttrain = transform(train_groups, :label => mean)
# train = select(ttrain, Not([:SEX, :MARRIAGE]))
# rename!(train, :label_mean => :group_weight)
# train = train[:, [1:21..., 23, 22]]

include("../../../DataUtils.jl")

categorical_features = coerce(select(df, [:EDUCATION, :MARRIAGE]), autotype(select(df, [:EDUCATION, :MARRIAGE]), rules=(:few_to_finite,)))
onehot_tensors = map(x -> Flux.onehotbatch(x, unique(x)), eachcol(categorical_features))
onehot_features = Matrix{Float32}(vcat(onehot_tensors...))

# categories = map(unique, eachcol(categorical_features))

# num_categories = map(lastindex, categories)

# categorical_features = mapcols(to_integer_categorical_vector, categorical_features)

continuous_features = select(df, 1, 5, 6:24)

df = hcat(binary_features, DataFrame(permutedims(onehot_features), :auto), continuous_features)

println(describe(df))

train, test = split_data(df; at=0.5)

#UnitRangeTransform

unit_transform_features_train_fits = map(col -> StatsBase.fit(UnitRangeTransform, Float32.(col)), eachcol(select(train, 13:14, 27:32)))

unit_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(train, 13:14, 27:32)))

unit_transform_features_train = DataFrame(unit_transform_features_train, vcat(names(train)[13:14], names(train)[27:32]))

#ZScoreTransform

z_transform_features_train_fits = map(col -> StatsBase.fit(ZScoreTransform, Float32.(col)), eachcol(select(train, 15:26)))

z_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), z_transform_features_train_fits, eachcol(select(train, 15:26)))

z_transform_features_train = DataFrame(z_transform_features_train, names(train)[15:26])

continuous_features_train = hcat(train[!, 1:12], unit_transform_features_train, z_transform_features_train)
# continuous_features_train.SEX = train[!, :SEX]
continuous_features_train.label = train[!, :label]

#### test

#UnitRangeTransform

unit_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(test, 13:14, 27:32)))

unit_transform_features_test = DataFrame(unit_transform_features_test, vcat(names(train)[13:14], names(train)[27:32]))

#ZScoreTransform


z_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), z_transform_features_train_fits, eachcol(select(test, 15:26)))

z_transform_features_test = DataFrame(z_transform_features_test, names(test)[15:26])

continuous_features_test = hcat(test[!, 1:12], unit_transform_features_test, z_transform_features_test)
# continuous_features_test.SEX = test[!, :SEX]
continuous_features_test.label = test[!, :label]

# categorical_features_train = coerce(select(train, [:EDUCATION, :MARRIAGE]), autotype(select(train, [:EDUCATION, :MARRIAGE]), rules=(:few_to_finite,)))
# categorical_features_test = coerce(select(test, [:EDUCATION, :MARRIAGE]), autotype(select(test, [:EDUCATION, :MARRIAGE]), rules=(:few_to_finite,)))





# a = mapcols(col -> StatsBase.standardize(UnitRangeTransform, Float32.(col)), select(train, 1, 5))
# b = mapcols(col -> StatsBase.standardize(ZScoreTransform, Float32.(col)), select(train, 6:17))
# c = mapcols(col -> StatsBase.standardize(ZScoreTransform, Float32.(col); center=false), select(train, 18:23))

# using MLJ
# continuous_features_train = hcat(a, b, c)
# categorical_features_train = coerce(select(train, [:SEX, :EDUCATION, :MARRIAGE]), autotype(select(train, [:SEX, :EDUCATION, :MARRIAGE]), rules=(:few_to_finite,)))
# continuous_features_train.label = train.label


# # # train_keys=keys(train_groups)
# # combines_train = combine(train_groups, :label => mean)
# # combines_train = combine(train_groups, :label => lastindex)

# a = mapcols(col -> StatsBase.standardize(UnitRangeTransform, Float32.(col)), select(test, 1, 5))
# b = mapcols(col -> StatsBase.standardize(ZScoreTransform, Float32.(col)), select(test, 6:17))
# c = mapcols(col -> StatsBase.standardize(ZScoreTransform, Float32.(col); center=false), select(test, 18:23))

# using MLJ
# continuous_features_test = hcat(a, b, c)
# categorical_features_test = coerce(select(test, [:SEX, :EDUCATION, :MARRIAGE]), autotype(select(test, [:SEX, :EDUCATION, :MARRIAGE]), rules=(:few_to_finite,)))
# continuous_features_test.label = test.label


# CSV.write("data.csv", df)

# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)

train_size = size(train, 1)
test_size = size(test, 1)
_size = min(train_size, test_size)
n_folds = 10
train_fold_size = minimum([1000, div(train_size, n_folds)])
test_fold_size = minimum([1000, div(test_size, n_folds)])

mkpath("./TenFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
    train = continuous_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # cat_train = categorical_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # # train, leftovers = balance_binary_data(train)
    # test = continuous_features_test[fold_size*i+1:fold_size*(i+1), :]
    # cat_test = categorical_features_test[fold_size*i+1:fold_size*(i+1), :]

    test = continuous_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # cat_test = categorical_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # # test = vcat(test, leftovers)
    CSV.write("./TenFolds/train_$(i).csv", train)
    # CSV.write("./TenFolds/categorical_train_$(i).csv", cat_train)
    CSV.write("./TenFolds/test_$(i).csv", test)
    # CSV.write("./TenFolds/categorical_test_$(i).csv", cat_test)
    # println((fold_size*(i-1))+1:fold_size*i)
    # println(1000*i+1:1000*(i+1))
end