using DataFrames, DelimitedFiles, CSV
using StatsBase
PATH = @__DIR__
cd(PATH)

df = CSV.read("creditcard.csv", DataFrame)

println(describe(df))
select!(df, Not([:Time, :Amount]))
rename!(df, "Class" => :label)
df[df.label.==0, :label] .= 2

CSV.write("data.csv", df)



# train, test = split_data(df)
# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)


Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]
train_size = 1000
test_size = 1000
n_folds = 10
fold_size = minimum([1000, div(size(df, 1), n_folds)])

mkpath("./TenFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
    train = df[(fold_size*(i-1))+1:fold_size*i, :]
    # # # train, leftovers = balance_binary_data(train)
    test = df[(fold_size*mod(i, n_folds))+1:fold_size*mod1((i + 1), n_folds), :]
    # # # test = vcat(test, leftovers)
    CSV.write("./TenFolds/train_$(i).csv", train)
    CSV.write("./TenFolds/test_$(i).csv", test)
    # println((fold_size*(i-1))+1:fold_size*i)
    # println((fold_size*mod(i, n_folds))+1:fold_size*mod1((i+1), n_folds))
end