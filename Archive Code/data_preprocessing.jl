using DataFrames, DelimitedFiles, Statistics

X = readdlm("./secom_data/secom_data.txt", ' ', Float32)
X = DataFrame(X, :auto)
total_samples = size(X)[1]
X = X[:, mean.(isnan, eachcol(X)).<0.55]
X = X[mean.(isnan, eachrow(X)).<0.8, :]
sum(count.(isnan, eachcol(X)))

for col in eachcol(X)
    if unique(col) == 2
        replace!(col, NaN => 0.0)
    elseif unique(col) == 3
        replace!(col, NaN => 0.0)
    else
        nothing
    end
end

for col in eachcol(X)
    if length(unique(col)) .< 10
        replace!(col, NaN => 0.0)
    else
        nothing
    end
end

for col in eachcol(X)
    if count(isnan, col) .> 0.1 * total_samples
        replace!(col, NaN => 0.0)
    else
        nothing
    end
end

sum(count.(isnan, eachcol(X)))


for col in eachcol(X)
    nan_indices = findall(isnan, col)
    for i in nan_indices
        if col[mod1(i - 1, total_samples)] != NaN && col[mod1(i + 1, total_samples)] != NaN
            col[i] = (col[mod1(i - 1, total_samples)] + col[mod1(i + 1, total_samples)]) / 2
        end
    end
end

for col in eachcol(X)
    replace!(col, NaN => 0.0)
end

sum(count.(isnan, eachcol(X)))

X = X[:, length.([unique(i) for i in eachcol(X)]).!=1]

Y = readdlm("./secom_data/secom_labels.txt", ' ')[:, 1]
Y[Y.==-1] .= 2
Y= Int.(Y)
Y = DataFrame([Y], [:target])

# writedlm("./secom_data/secom_data_preprocessed_moldovan2017.csv", Matrix(X), ',')