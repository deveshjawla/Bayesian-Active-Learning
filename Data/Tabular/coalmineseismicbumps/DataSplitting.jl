using ARFFFiles
using StatsBase
using DataFrames, DelimitedFiles, CSV, MLJ, Flux
PATH = @__DIR__
cd(PATH)
include("../../../DataUtils.jl")


df = ARFFFiles.load(DataFrame, "seismic-bumps.arff")

println(describe(df))

map(x -> lastindex(unique(x)), eachcol(select(df, [:seismic, :shift, :seismoacoustic, :ghazard])))

binary_features = mapcols(to_binary_vector, select(df, [:seismic, :shift]))

categorical_features = coerce(select(df, [:seismoacoustic, :ghazard]), autotype(select(df, [:seismoacoustic, :ghazard]), rules=(:few_to_finite,)))

onehot_tensors = map(x -> Flux.onehotbatch(x, unique(x)), eachcol(categorical_features))
onehot_features = Matrix{Float32}(vcat(onehot_tensors...))

# shift = Int.(permutedims(unique(df.shift)) .== df.shift)
# seismic = Int.(permutedims(unique(df.seismic)) .== df.seismic)
# seismoacoustic = Int.(permutedims(unique(df.seismoacoustic)) .== df.seismoacoustic)
# ghazard = Int.(permutedims(unique(df.ghazard)) .== df.ghazard)

# insertcols!(df, :shift, :shift => shift[:, 1], :shift => shift[:, 2], makeunique=true, after=true)
# insertcols!(df, :seismic, :seismic => seismic[:, 1], :seismic => seismic[:, 2], makeunique=true, after=true)
# insertcols!(df, :seismoacoustic, :seismoacoustic => seismoacoustic[:, 1], :seismoacoustic => seismoacoustic[:, 2], makeunique=true, after=true)
# insertcols!(df, :ghazard, :ghazard => ghazard[:, 1], :ghazard => ghazard[:, 2], makeunique=true, after=true)

rename!(df, :class => :label)
df[!, :label] = Int.("1" .== df.label)
df[df.label.==0, :label] .= 2


continuous_features = select(df, Not([:seismic, :shift, :seismoacoustic, :ghazard]))


df = hcat(binary_features, DataFrame(permutedims(onehot_features), :auto), continuous_features)
println(describe(df))

select!(df, Not(:nbumps6, :nbumps7, :nbumps89))


# train, test = split_data(float_features)
# train = balance_binary_data(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
# # test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

# CSV.write("./train.csv", train)
# CSV.write("./test.csv", test)
Random.seed!(1234)
df = df[shuffle(axes(df, 1)), :]

df_size, = size(df, 1)
n_folds = 10
fold_size = minimum([1000, div(df_size, n_folds)])

fold_incides = cross_validation_indices(df_size, n_folds)

mkpath("./TenFolds")
#generate five folds and save them as train/test split in the 5 Folds Folder
for i in 1:n_folds
    # train = continuous_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # # cat_train = categorical_features_train[(train_fold_size*(i-1))+1:train_fold_size*i, :]
    # # # train, leftovers = balance_binary_data(train)
    # # test = continuous_features_test[fold_size*i+1:fold_size*(i+1), :]
    # # cat_test = categorical_features_test[fold_size*i+1:fold_size*(i+1), :]

    # test = continuous_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # # cat_test = categorical_features_test[(test_fold_size*(i-1))+1:test_fold_size*i, :]
    # # # test = vcat(test, leftovers)

    train_indices, test_indices = fold_incides[i]
    train = df[train_indices, :]
    test = df[test_indices, :]

    #UnitRangeTransform

    unit_transform_features_train_fits = map(col -> StatsBase.fit(UnitRangeTransform, Float32.(col)), eachcol(select(train, 9:10, 13:19)))

    unit_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(train, 9:10, 13:19)))

    unit_transform_features_train = DataFrame(unit_transform_features_train, vcat(names(train)[9:10], names(train)[13:19]))

    #ZScoreTransform

    z_transform_features_train_fits = map(col -> StatsBase.fit(ZScoreTransform, Float32.(col)), eachcol(select(train, 11:12)))

    z_transform_features_train = map((t, col) -> StatsBase.transform(t, Float32.(col)), z_transform_features_train_fits, eachcol(select(train, 11:12)))

    z_transform_features_train = DataFrame(z_transform_features_train, names(train)[11:12])

    continuous_features_train = hcat(train[!, 1:8], unit_transform_features_train, z_transform_features_train)
    # continuous_features_train.SEX = train[!, :SEX]
    continuous_features_train.label = train[!, :label]

    #### test

    #UnitRangeTransform

    unit_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), unit_transform_features_train_fits, eachcol(select(test, 9:10, 13:19)))

    unit_transform_features_test = DataFrame(unit_transform_features_test, vcat(names(train)[9:10], names(train)[13:19]))

    #ZScoreTransform


    z_transform_features_test = map((t, col) -> StatsBase.transform(t, Float32.(col)), z_transform_features_train_fits, eachcol(select(test, 11:12)))

    z_transform_features_test = DataFrame(z_transform_features_test, names(test)[11:12])

    continuous_features_test = hcat(test[!, 1:8], unit_transform_features_test, z_transform_features_test)
    # continuous_features_test.SEX = test[!, :SEX]
    continuous_features_test.label = test[!, :label]


    map(x -> replace!(x, NaN => 0.0), eachcol(continuous_features_train))
    map(x -> replace!(x, NaN => 0.0), eachcol(continuous_features_test))

    CSV.write("./TenFolds/train_$(i).csv", continuous_features_train)
    # CSV.write("./TenFolds/categorical_train_$(i).csv", cat_train)
    CSV.write("./TenFolds/test_$(i).csv", continuous_features_test)
    # CSV.write("./TenFolds/categorical_test_$(i).csv", cat_test)
    # println((fold_size*(i-1))+1:fold_size*i)
    # println(1000*i+1:1000*(i+1))
end




# using MultivariateStats

# M = fit(KernelPCA, permutedims(Matrix(select(train, Not([:label])))), maxoutdim=2)
# train_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(train, Not([:label])))))

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, permutedims(Matrix(select(test, Not([:label])))))

# CSV.write("../coalmineKernelPCA/train.csv", DataFrame(hcat(permutedims(train_x_transformed), train.label), :auto))
# CSV.write("../coalmineKernelPCA/test.csv", DataFrame(hcat(permutedims(test_x_transformed), test.label), :auto))