using DataFrames, DelimitedFiles, CSV

PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")

df = CSV.read("./data_banknote_authentication.txt", DataFrame, header=false)
rename!(df, :Column5 => :label)
df[df.label.==0, :label] .= 2

# train, test = split_data(df)
# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)

Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 40
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

