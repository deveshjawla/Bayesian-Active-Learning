using DataFrames, DelimitedFiles, CSV

PATH = @__DIR__
cd(PATH)

train= CSV.read("train.csv", DataFrame, header=1)
test= CSV.read("test.csv", DataFrame, header=1)
validate= CSV.read("validate.csv", DataFrame, header=1)
train = vcat(train, validate)

using MultivariateStats

M = fit(PCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim = 3)
train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# M = fit(PCA, test_x', maxoutdim = 150)
test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))


df = vcat(train, test)
rename!(df, :stroke => :label)
df[df.label.==0, :label] .= 2
using Random
function train_validate_test(df; v=0.6, t=0.8)
	    r = size(df, 1)
	    val_index = Int(round(r * v))
	    test_index = Int(round(r * t))
		df=df[shuffle(axes(df, 1)), :]
	    train = df[1:val_index, :]
	    validate = df[(val_index+1):test_index, :]
	    test = df[(test_index+1):end, :]
	    return train, validate, test
	end

	train,test,validate=train_validate_test(df)
	train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	validate = data_balancing(validate, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	

	CSV.write("../strokePCA_dataset/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
	CSV.write("../strokePCA_dataset/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))
	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
	CSV.write("./validate.csv", validate)
	