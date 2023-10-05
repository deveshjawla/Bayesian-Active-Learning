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
train,test=split_data(df)
CSV.write("./train.csv", train)
CSV.write("./test.csv", test)

groups = groupby(df, :label)
for (k, g) in pairs(groups)
	group_name = k.label
	group = select(g, Not(:label))
	train,test=split_data(group)
	mkpath("./Groups")
	CSV.write("./Groups/$(group_name)_train.csv", train)
	CSV.write("./Groups/$(group_name)_test.csv", test)
end
group_names=collect.(keys(groups))
writedlm("./Groups/group_names.txt", group_names, ',')