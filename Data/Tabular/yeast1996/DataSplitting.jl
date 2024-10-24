using DataFrames, DelimitedFiles, CSV, MLJ
PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")
df = CSV.read("./data.csv", DataFrame, header=false; stripwhitespace=true)
rename!(df, :Column10 => :label)

labels = Vector{Any}(df.label)
for (i, j) in enumerate(unique(labels))
    replace!(labels, j => i)
end
labels = Int.(labels)

df.label = labels

df = select(df, Not(:Column1))
# train, test = split_data(df; at=0.5)
# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)
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
    CSV.write("./TenFolds/train_$(i).csv", train)
    # CSV.write("./TenFolds/categorical_train_$(i).csv", cat_train)
    CSV.write("./TenFolds/test_$(i).csv", test)
    # CSV.write("./TenFolds/categorical_test_$(i).csv", cat_test)
    # println((fold_size*(i-1))+1:fold_size*i)
    # println(1000*i+1:1000*(i+1))
end

# groups = groupby(df, :label)
# for (k, g) in pairs(groups)
# 	group_name = k.label
# 	group = select(g, Not(:label))
# 	train,test=split_data(group)
# 	mkpath("./Groups")
# 	CSV.write("./Groups/$(group_name)_train.csv", train)
# 	CSV.write("./Groups/$(group_name)_test.csv", test)
# end
# group_names=collect.(keys(groups))
# writedlm("./Groups/group_names.csv", group_names, ',')