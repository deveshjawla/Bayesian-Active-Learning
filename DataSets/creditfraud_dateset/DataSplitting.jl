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

	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
	CSV.write("./validate.csv", validate)
	