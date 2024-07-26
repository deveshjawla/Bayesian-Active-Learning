using DataFrames, DelimitedFiles, CSV
PATH = @__DIR__
cd(PATH)
df=CSV.read("./data.csv", DataFrame, header=1)
rename!(df, :Rings => :label)
df.label = minmaxscaling(df.label, 29, 1)
groups = groupby(df, :Sex)
include("../../../../DataUtils.jl")
for (k, g) in pairs(groups)
	# group_name = k.Sex
	group = select(g, Not(:Sex))
	# println(describe(g))
	# group.label = minmaxscaling(group.label, 29, 1)
	println(mean(group.label))
	# group[!, :group_weight] = ones(size(group, 1)) * mean(group.label)
end
tdf = transform(groups, :label => mean)
tdf.Sex = tdf.label_mean
names=collect.(keys(groups))
writedlm("./Groups/names.csv", names, ',')

train,test=split_data(group)
mkpath("./Groups")
CSV.write("./Groups/$(group_name)_train.csv", train)
CSV.write("./Groups/$(group_name)_test.csv", test)

select!(tdf, Not(:label_mean))
train,test=split_data(df)
CSV.write("./train.csv", train)
CSV.write("./test.csv", test)