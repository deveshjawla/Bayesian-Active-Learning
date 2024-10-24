using DataFrames, DelimitedFiles, CSV, Statistics, MLJ, StatsBase, Flux
PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")

train = CSV.read("./data.txt", DataFrame, header=1, normalizenames=true; stripwhitespace=true)
test = CSV.read("./test.txt", DataFrame, header=1, normalizenames=true; stripwhitespace=true)

df = vcat(train, test)

println(describe(df))
println(map(x -> lastindex(unique(x)), eachcol(select(df, [:race, :occupation, :relationship, :workclass, :marital_status]))))
println(map(x -> countmap(x), eachcol(select(df, [:native_country]))))

df.native_country = df.native_country .== "United-States"

binary_features = mapcols(to_binary_vector, select(df, [:sex, :native_country]))
categorical_features = coerce(select(df, [:race, :occupation, :relationship, :workclass, :marital_status]), autotype(select(df, [:race, :occupation, :relationship, :workclass, :marital_status]), rules=(:few_to_finite,)))
onehot_tensors = map(x -> Flux.onehotbatch(x, unique(x)), eachcol(categorical_features))
onehot_features = Matrix{Float32}(vcat(onehot_tensors...))

# categories = map(unique, eachcol(categorical_features))

# num_categories = map(lastindex, categories)

# categorical_features = mapcols(to_integer_categorical_vector, categorical_features)

df.label = df.label .== ">50K"
df.label = replace(df.label, false => 2)

continuous_features = select(df, [:age, :education_num, :hour_per_week, :fnlwgt, :capital_gain, :capital_loss, :label])

df = hcat(binary_features, DataFrame(permutedims(onehot_features), :auto), continuous_features)
println(describe(df))

train, test = split_data(df; at=0.5)

unit_transform_features_train_fits = map(col -> StatsBase.fit(UnitRangeTransform, Float32.(col)), eachcol(select(train, 45:50)))

unit_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(train, 45:50)))

unit_transform_features_train = DataFrame(unit_transform_features_train, names(train)[45:50])

continuous_features_train = hcat(train[!, 1:44], unit_transform_features_train)
# continuous_features_train.sex = train[!, :sex]
# continuous_features_train.native_country = train[!, :native_country]
continuous_features_train.label = train[!, :label]

unit_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(test, 45:50)))

unit_transform_features_test = DataFrame(unit_transform_features_test, names(test)[45:50])

continuous_features_test = hcat(test[!, 1:44], unit_transform_features_test)
# continuous_features_test.sex = test[!, :sex]
# continuous_features_test.native_country = test[!, :native_country]
continuous_features_test.label = test[!, :label]

# categorical_features_train = coerce(select(train, [:race, :occupation, :relationship, :workclass, :marital_status]), autotype(select(train, [:race, :occupation, :relationship, :workclass, :marital_status]), rules=(:few_to_finite,)))
# categorical_features_test = coerce(select(test, [:race, :occupation, :relationship, :workclass, :marital_status]), autotype(select(test, [:race, :occupation, :relationship, :workclass, :marital_status]), rules=(:few_to_finite,)))



# train = coerce(train, autotype(train, rules=(:discrete_to_continuous, :few_to_finite)))

# # train.marital_status = train.marital_status .== "Never-married"
# train.native_country = train.native_country .== "United-States"
# train.race = train.race .== "White"
# train_groups = groupby(train, [:workclass, :marital_status, :education, :relationship, :sex, :occupation, :race, :native_country])
# train_keys = keys(train_groups)
# combines_train = combine(train_groups, :label => mean)
# group_weights_list = combines_train.label_mean
# train_keys = Dict(keys(train_groups) .=> group_weights_list)

# CSV.write("./combi.csv",combines_train)

# train_groups = groupby(train, [:sex])
# train_groups = groupby(train, [:race])
# train_groups = groupby(train, [:occupation])
# train_groups = groupby(train, [:native_country])
# for group in train_groups
#     println(mean(group.label))
#     println("group size is   ", size(group))
# end
# ttrain = transform(train_groups, :label => mean)
# # train = select(ttrain, Not([:sex, :occupation, :race, :native_country]))
# # rename!(train, :label_mean => :group_weight)
# # train = select(train, [:group_weight, :age, :education_num, :hour_per_week, :label])



# # test = select(test, Not([:workclass, :education, :marital_status, :fnlwgt, :capital_gain, :capital_loss, :relationship]))
# # test = dropmissing(test)
# # test.age = minmaxscaling(test.age, 90, 17)
# # test.education_num = minmaxscaling(test.education_num, 16, 1)
# # test.hour_per_week = minmaxscaling(test.hour_per_week, 99, 1)
# test = dropmissing(test)
# test.age = minmaxscaling(test.age, maximum(test.age), minimum(test.age))
# test.education_num = minmaxscaling(test.education_num, maximum(test.education_num), minimum(test.education_num))
# test.hour_per_week = minmaxscaling(test.hour_per_week, maximum(test.hour_per_week), minimum(test.hour_per_week))
# test.fnlwgt = minmaxscaling(test.fnlwgt, maximum(test.fnlwgt), minimum(test.fnlwgt))
# test.capital_gain = minmaxscaling(test.capital_gain, maximum(test.capital_gain), minimum(test.capital_gain))
# test.capital_loss = minmaxscaling(test.capital_loss, maximum(test.capital_loss), minimum(test.capital_loss))


# continuous_features_train = coerce(select(train, [:age, :fnlwgt, :capital_gain, :capital_loss, :education_num, :hour_per_week, :label]), autotype(select(train, [:age, :fnlwgt, :capital_gain, :capital_loss, :education_num, :hour_per_week, :label]), rules=(:discrete_to_continuous,)))
# continuous_features_test = coerce(select(test, [:age, :fnlwgt, :capital_gain, :capital_loss, :education_num, :hour_per_week, :label]), autotype(select(test, [:age, :fnlwgt, :capital_gain, :capital_loss, :education_num, :hour_per_week, :label]), rules=(:discrete_to_continuous,)))
# test.marital_status = test.marital_status .== "Never-married"
# test.native_country = test.native_country .== "United-States"
# test.race = test.race .== "White"
# test_groups = groupby(test, [:sex, :occupation, :race, :native_country])
# test_keys = keys(test_groups)
# list_keys = intersect(train_keys, test_keys)
# mkpath("./Groups")
# writedlm("./Groups/list_keys.csv", list_keys, ',')

# empty_df = DataFrame()
# for (i, k) in enumerate(test_keys)
#     # train_group = select(train_groups[k], Not([:sex, :occupation, :race, :native_country]))
#     if k ∈ keys(train_keys)
#         test_group = select(test_groups[k], Not([:sex, :occupation, :race, :native_country]))
#         test_group[!, :group_weight] = ones(size(test_group, 1)) .* train_keys[k]
#         # println(describe(test_group))
#         empty_df = vcat(empty_df, test_group)
#     end
# end

# test = select(empty_df, [:group_weight, :age, :education_num, :hour_per_week, :label])



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