using ARFFFiles
using StatsBase
using DataFrames, DelimitedFiles, CSV
PATH = @__DIR__
cd(PATH)

df = ARFFFiles.load(DataFrame, "seismic-bumps.arff")

println(describe(df))

# shift = Int.(permutedims(unique(df.shift)) .== df.shift)
# seismic = Int.(permutedims(unique(df.seismic)) .== df.seismic)
# seismoacoustic = Int.(permutedims(unique(df.seismoacoustic)) .== df.seismoacoustic)
# ghazard = Int.(permutedims(unique(df.ghazard)) .== df.ghazard)

# insertcols!(df, :shift, :shift => shift[:, 1], :shift => shift[:, 2], makeunique=true, after=true)
# insertcols!(df, :seismic, :seismic => seismic[:, 1], :seismic => seismic[:, 2], makeunique=true, after=true)
# insertcols!(df, :seismoacoustic, :seismoacoustic => seismoacoustic[:, 1], :seismoacoustic => seismoacoustic[:, 2], makeunique=true, after=true)
# insertcols!(df, :ghazard, :ghazard => ghazard[:, 1], :ghazard => ghazard[:, 2], makeunique=true, after=true)

select!(df, Not([:seismic, :shift, :seismoacoustic, :ghazard]))

rename!(df, :class => :label)
df[!, :label] = Int.("1" .== df.label)
df[df.label.==0, :label] .= 2

a = mapcols(col-> StatsBase.standardize(UnitRangeTransform, Float64.(col)), select(df, 1,2,5:9,13,14))
b = mapcols(col-> StatsBase.standardize(ZScoreTransform, Float64.(col)), select(df, 3,4))

float_features = hcat(a, b)
float_features.label = df.label


include("../../../DataUtils.jl")
train, test = split_data(float_features)
train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

CSV.write("./train.csv", train)
CSV.write("./test.csv", test)


# using MultivariateStats

# M = fit(KernelPCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim=2)
# train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))

# CSV.write("../coalmineKernelPCA/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
# CSV.write("../coalmineKernelPCA/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))