using DataFrames, DelimitedFiles, CSV, Statistics
PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")

train=CSV.read("./data.txt", DataFrame, header=1, normalizenames=true; stripwhitespace=true)
# names(train)
train = select(train, Not([:workclass, :marital_status, :education, :fnlwgt, :capital_gain, :capital_loss, :relationship]))
train = dropmissing(train)
train.age = minmaxscaling(train.age, 90, 17)
train.education_num = minmaxscaling(train.education_num, 16, 1)
train.hour_per_week = minmaxscaling(train.hour_per_week, 99, 1)
train.label = train.label .== ">50K"
# train.marital_status = train.marital_status .== "Never-married"
train.native_country = train.native_country .== "United-States"
train.race = train.race .== "White"
train_groups = groupby(train, [:sex, :occupation, :race, :native_country])
# train_keys=keys(train_groups)
combines_train = combine(train_groups, :label=> mean)
group_weights_list = combines_train.label_mean
train_keys= Dict(keys(train_groups) .=> group_weights_list)

CSV.write("./combi.csv",combines_train)

# train_groups = groupby(train, [:sex])
# train_groups = groupby(train, [:race])
# train_groups = groupby(train, [:occupation])
# train_groups = groupby(train, [:native_country])
# for group in train_groups
# 	println(mean(group.label))
# 	println("group size is   ", size(group))
# end
ttrain = transform(train_groups, :label=>mean)
train = select(ttrain, Not([:sex, :occupation, :race, :native_country]))
rename!(train, :label_mean => :group_weight)
train = select(train, [:group_weight, :age, :education_num, :hour_per_week, :label])


test=CSV.read("./test.txt", DataFrame, header=1, normalizenames=true; stripwhitespace=true)
test = select(test, Not([:workclass, :education, :marital_status, :fnlwgt, :capital_gain, :capital_loss, :relationship]))
test = dropmissing(test)
test.age = minmaxscaling(test.age, 90, 17)
test.education_num = minmaxscaling(test.education_num, 16, 1)
test.hour_per_week = minmaxscaling(test.hour_per_week, 99, 1)
test.label = test.label .== ">50K."
# test.marital_status = test.marital_status .== "Never-married"
test.native_country = test.native_country .== "United-States"
test.race = test.race .== "White"
test_groups = groupby(test, [:sex, :occupation, :race, :native_country])
test_keys = keys(test_groups)
list_keys = intersect(train_keys, test_keys)
# mkpath("./Groups")
# writedlm("./Groups/list_keys.csv", list_keys, ',')

empty_df = DataFrame()
for (i, k) in enumerate(test_keys)
	# train_group = select(train_groups[k], Not([:sex, :occupation, :race, :native_country]))
	if k ∈ keys(train_keys)
	test_group = select(test_groups[k], Not([:sex, :occupation, :race, :native_country]))
	test_group[!, :group_weight] = ones(size(test_group, 1)) .* train_keys[k]
	# println(describe(test_group))
	empty_df = vcat(empty_df, test_group)
end
end

test = select(empty_df, [:group_weight, :age, :education_num, :hour_per_week, :label])

train.label = replace(train.label, false=>2)
test.label = replace(test.label, false=>2)

# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)


full_data = vcat(train, test)
Random.seed!(1234)
df = full_data[shuffle(axes(full_data, 1)), :]
train_size = 1000
test_size = 1000
n_folds = 10
fold_size = 1000 #div(size(df, 1), n_folds)

mkpath("./FiveFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
	train = df[(fold_size*(i-1))+1:fold_size*i, :]
	# # train, leftovers = balance_binary_data(train)
	test = df[fold_size*i+1:fold_size*(i+1), :]
	# # test = vcat(test, leftovers)
	CSV.write("./FiveFolds/train_$(i).csv", train)
	CSV.write("./FiveFolds/test_$(i).csv", test)
	# println((fold_size*(i-1))+1:fold_size*i)
	# println(1000*i+1:1000*(i+1))
end