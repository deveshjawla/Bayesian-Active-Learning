using DataFrames, DelimitedFiles, CSV

PATH = @__DIR__
cd(PATH)

df=CSV.read("Iris_cleaned.csv", DataFrame, header=1)
rename!(df, :Species => :label)
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


	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
	CSV.write("./validate.csv", validate)
	