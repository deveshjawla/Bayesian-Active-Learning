using DataFrames, DelimitedFiles, CSV, StatsBase, MLJ, Flux

PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")
df = CSV.read("healthcare-dataset-stroke-data.csv", DataFrame, header=1)
println(describe(df))
# mapcols(col -> println(unique(col)), select(df, ["hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]))
rename!(df, :stroke => :label)

# df_no_bmi = df[df.bmi.=="N/A", :]
# df_bmi = df[df.bmi.!="N/A", :]
# CSV.write("./data_no_bmi.csv", df_no_bmi)
# CSV.write("./data_bmi.csv", df_bmi)

df = CSV.read("data_bmi.csv", DataFrame, header=1)
println(describe(df))

println(map(x -> lastindex(unique(x)), eachcol(select(df, [:gender, :smoking_status]))))

binary_features = mapcols(to_binary_vector, select(df, [:hypertension, :heart_disease, :ever_married, :Residence_type]))

categorical_features = coerce(select(df, [:gender, :smoking_status, :work_type]), autotype(select(df, [:gender, :smoking_status, :work_type]), rules=(:few_to_finite,)))
onehot_tensors = map(x -> Flux.onehotbatch(x, unique(x)), eachcol(categorical_features))
onehot_features = Matrix{Float32}(vcat(onehot_tensors...))


# categories = map(unique, eachcol(categorical_features))

# num_categories = map(lastindex, categories)

# categorical_features = mapcols(to_integer_categorical_vector, categorical_features)

df.label = replace(df.label, 0 => 2)

continuous_features = select(df, [:age, :avg_glucose_level, :bmi, :label])

df = hcat(binary_features, DataFrame(permutedims(onehot_features), :auto), continuous_features)
println(describe(df))


# categorical_features_train = coerce(select(train, [:gender, :smoking_status]), autotype(select(train, [:gender, :smoking_status]), rules=(:few_to_finite,)))
# categorical_features_test = coerce(select(test, [:gender, :smoking_status]), autotype(select(test, [:gender, :smoking_status]), rules=(:few_to_finite,)))



# train_groups = groupby(train, ["hypertension", "heart_disease", "ever_married"]) #"gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"
# combines_train = combine(train_groups, :label => mean)
# group_weights_list = combines_train.label_mean
# train_keys = Dict(keys(train_groups) .=> group_weights_list)

# ttrain = transform(train_groups, :label => mean)
# train = select(ttrain, Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
# rename!(train, :label_mean => :group_weight)
# train = select(train, [:group_weight, :age, :avg_glucose_level, :bmi, :label])
# # continuos_data = select(df_bmi, Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
# train[train.label.==0, :label] .= 2

# test_groups = groupby(test, ["hypertension", "heart_disease", "ever_married"])
# test_keys = keys(test_groups)
# #"gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"
# empty_df = DataFrame()
# for (i, k) in enumerate(test_keys)
#     # train_group = select(train_groups[k], Not([:sex, :occupation, :race, :native_country]))
#     if k ∈ keys(train_keys)
#         test_group = select(test_groups[k], Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
#         test_group[!, :group_weight] = ones(size(test_group, 1)) .* train_keys[k]
#         # println(describe(test_group))
#         empty_df = vcat(empty_df, test_group)
#     end
# end
# test = select(empty_df, [:group_weight, :age, :avg_glucose_level, :bmi, :label])
# test[test.label.==0, :label] .= 2


# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)


# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)


# using MultivariateStats

# M = fit(PCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim = 3)
# train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))


# CSV.write("../strokePCA/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
# CSV.write("../strokePCA/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))

Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]

df_size, = size(df, 1)
n_folds = 10
fold_size = minimum([1000, div(df_size, n_folds)])

fold_incides = cross_validation_indices(df_size, n_folds)

mkpath("./TenFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
    # train = continuous_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # # cat_train = categorical_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # # # train, leftovers = balance_binary_data(train)
    # # test = continuous_features_test[fold_size*i+1:fold_size*(i+1), :]
    # # cat_test = categorical_features_test[fold_size*i+1:fold_size*(i+1), :]

    # test = continuous_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # # cat_test = categorical_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # # # test = vcat(test, leftovers)

    train_indices, test_indices = fold_incides[i]
    train = df[train_indices, :]
    test = df[test_indices, :]

    #UnitRangeTransform

    unit_transform_features_train_fits = map(col -> StatsBase.fit(UnitRangeTransform, Float32.(col)), eachcol(select(train, 17:19)))

    unit_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(train, 17:19)))

    unit_transform_features_train = DataFrame(unit_transform_features_train, names(train)[17:19])

    continuous_features_train = hcat(train[!, 1:16], unit_transform_features_train)

    continuous_features_train.label = train[!, :label]


    unit_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(test, 17:19)))

    unit_transform_features_test = DataFrame(unit_transform_features_test, names(test)[17:19])

    continuous_features_test = hcat(test[!, 1:16], unit_transform_features_test)

    continuous_features_test.label = test[!, :label]


    CSV.write("./TenFolds/train_$(i).csv", continuous_features_train)
    # CSV.write("./TenFolds/categorical_train_$(i).csv", cat_train)
    CSV.write("./TenFolds/test_$(i).csv", continuous_features_test)
    # CSV.write("./TenFolds/categorical_test_$(i).csv", cat_test)
    # println((fold_size*(i-1))+1:fold_size*i)
    # println(1000*i+1:1000*(i+1))
end