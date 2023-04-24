using Random

function data_balancing(data_xy; balancing::String, positive_class_label=1, negative_class_label=2)
    negative_class = data_xy[data_xy[:, end].==negative_class_label, :]
    positive_class = data_xy[data_xy[:, end].==positive_class_label, :]
    size_positive_class = size(positive_class)[1]
    size_normal = size(negative_class)[1]
    multiplier = div(size_normal, size_positive_class)
    leftover = mod(size_normal, size_positive_class)
    if balancing == "undersampling"
        data_xy = vcat(negative_class[1:size(positive_class)[1], :], positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "generative"
        new_positive_class = vcat(repeat(positive_class, outer=multiplier - 1), positive_class[1:leftover, :], positive_class)
        data_x = select(new_positive_class, Not([:target]))
        data_y = select(new_positive_class, [:target])
        new_positive_class = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_positive_class = hcat(data_x, data_y)
        data_xy = vcat(negative_class, new_positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "none"
        nothing
    end
    # data_x = Matrix(data_xy)[:, 1:end-1]
    # data_y = data_xy.target
    return data_xy
end

# A handy helper function to normalize our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ 0.000001)
end

# A handy helper function to normalize our dataset.
function scaling(x, max_, min_)
    return (x .- min_) ./ (max_ - min_)
end

function pool_test_maker(pool, test, n_input)
    pool = Matrix(permutedims(pool))
    test = Matrix(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
    # pool_max = maximum(pool_x, dims=1)
    # pool_mini = minimum(pool_x, dims=1)
    # pool_x = scaling(pool_x, pool_max, pool_mini)
    pool_mean = mean(pool_x, dims=2)
    pool_std = std(pool_x, dims=2)
    pool_x = standardize(pool_x, pool_mean, pool_std)

    test_x = test[1:n_input, :]
    test_y = test[end, :]
    # test_x = scaling(test_x, pool_max, pool_mini)
    test_x = standardize(test_x, pool_mean, pool_std)


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end


# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    index = Int(round(r * at))
    pool = df[1:index, :]
    test = df[(index+1):end, :]
    return pool, test
end