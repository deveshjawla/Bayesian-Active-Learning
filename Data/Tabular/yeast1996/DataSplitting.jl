using DataFrames, DelimitedFiles, CSV
PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")
df=CSV.read("./data.csv", DataFrame, header=false; stripwhitespace=true)
rename!(df, :Column10 => :label)

labels = Vector{Any}(df.label)
for (i,j) in enumerate(unique(labels))
	replace!(labels, j=>i)
end
labels = Int.(labels)

df.label = labels

df = select(df, Not(:Column1))
# train,test=split_data(df)
# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)
Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 296
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