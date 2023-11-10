using DataFrames, DelimitedFiles, CSV, StatsBase

PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")
df = CSV.read("healthcare-dataset-stroke-data.csv", DataFrame, header=1)
println(describe(df))
mapcols(col -> println(unique(col)), select(df, ["hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]))
rename!(df, :stroke => :label)

df_no_bmi = df[df.bmi.=="N/A", :]
df_bmi = df[df.bmi.!="N/A", :]
CSV.write("./data_no_bmi.csv", df_no_bmi)
CSV.write("./data_bmi.csv", df_bmi)

df = CSV.read("data_bmi.csv", DataFrame, header=1)
println(describe(df_bmi))
df.age = minmaxscaling(df.age, 90, 0.08)
df.avg_glucose_level = minmaxscaling(df.avg_glucose_level, 300, 50)
df.bmi = minmaxscaling(df.bmi, 100, 10)
train, test = split_data(df)

train_groups = groupby(train, ["hypertension", "heart_disease", "ever_married"]) #"gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"
combines_train = combine(train_groups, :label=> mean)
group_weights_list = combines_train.label_mean
train_keys= Dict(keys(train_groups) .=> group_weights_list)

ttrain = transform(train_groups, :label=>mean)
train = select(ttrain, Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
rename!(train, :label_mean => :group_weight)
train = select(train, [:group_weight, :age, :avg_glucose_level,  :bmi, :label])
# continuos_data = select(df_bmi, Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
train[train.label.==0, :label] .= 2

test_groups = groupby(test, ["hypertension", "heart_disease", "ever_married"]) 
test_keys = keys(test_groups)
#"gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"
empty_df = DataFrame()
for (i, k) in enumerate(test_keys)
	# train_group = select(train_groups[k], Not([:sex, :occupation, :race, :native_country]))
	if k ∈ keys(train_keys)
	test_group = select(test_groups[k], Not(["id", "gender", "ever_married", "work_type", "Residence_type", "smoking_status", "hypertension", "heart_disease"]))
	test_group[!, :group_weight] = ones(size(test_group)[1]) .* train_keys[k]
	# println(describe(test_group))
	empty_df = vcat(empty_df, test_group)
end
end
test = select(empty_df, [:group_weight, :age, :avg_glucose_level,  :bmi, :label])
test[test.label.==0, :label] .= 2


# train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)


# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)


# using MultivariateStats

# M = fit(PCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim = 3)
# train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))


# CSV.write("../strokePCA/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
# CSV.write("../strokePCA/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))

df = vcat(train, test)
Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 40
n_folds = 5
fold_size = div(size(df)[1], n_folds)

mkpath("./FiveFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
	train = df[(fold_size*(i-1))+1:fold_size*i, :]
	train, leftovers = balanced_binary_maker(train, positive_class_label=1, negative_class_label=2, maximum_per_class = Int(train_size/2))
	test = df[Not((fold_size*(i-1))+1:fold_size*i), :]
	test = vcat(test, leftovers)
	CSV.write("./FiveFolds/train_$(i).csv", train)
	CSV.write("./FiveFolds/test_$(i).csv", test)
	# println((fold_size*(i-1))+1:fold_size*i)
end
