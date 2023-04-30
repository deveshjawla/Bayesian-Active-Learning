using DelimitedFiles, DataFrames, Plots
using Statistics

## -------------- load data  -------------- ##
X = readdlm("./secom_data/secom_data_preprocessed_moldovan2017.csv", ',', Float32)

corr_matrix = cor(X)

using LinearAlgebra

lr_corr_matrix = LowerTriangular(corr_matrix)

for (i, j) in zip(1:356, 1:356)
    lr_corr_matrix[i, j] = 0.0
end

pos_corr = findall(>(0.95), lr_corr_matrix)

num_corr = length(pos_corr)
function cartesian_list(num_corr)
    feature_indices = Vector{Int}(undef, 2 * num_corr)

    for i in 1:num_corr
        feature_indices[i] = pos_corr[i][1]
        feature_indices[i+num_corr] = pos_corr[i][2]
    end
    return feature_indices
end

feature_indices = cartesian_list(num_corr)

using StatsBase

sorted_feature_indices = countmap(feature_indices[:,1])

df = DataFrame(:i => collect(keys(sorted_feature_indices)), :j => collect(values(sorted_feature_indices)))

df = sort(df, :j)
df[1:83, :]
X = X[:, Not(df[1:83, 1])]

writedlm("removed_95pct_corr_features.csv", X, ',')