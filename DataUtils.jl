"""
Generates three Gaussian(0,1) Distributed blobs around (0,0), (2,2) and (-2,2)
and labels for each blob

Input: n = the number of points needed per blob(class)
Output: Tuple{Matrix, Matrix} = Input features X (coordinates in 2d space), Onehot labels of the blobs
"""
function gen_3_clusters(n; cluster_centers=[[0, 0], [2, 2], [-2, 2]])
    x1 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[1]
    x2 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[2]
    x3 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_centers[3]
    y1 = vcat(ones(Float32, n), zeros(Float32, 2 * n))
    y2 = vcat(zeros(Float32, n), ones(Float32, n), zeros(Float32, n))
    y3 = vcat(zeros(Float32, n), zeros(Float32, n), ones(Float32, n))
    return hcat(x1, x2, x3), permutedims(hcat(y1, y2, y3))
end

function gen_1_clusters(n; cluster_center=[2, -1])
    x1 = randn(Xoshiro(1234), Float32, 2, n) .+ cluster_center
    y1 = ones(Float32, n)
    return hcat(x1), permutedims(y1)
end

function pairs_to_matrix(X1, X2)
    n_pairs = lastindex(X1) * lastindex(X2)
    test_x_area = Matrix{Float32}(undef, 2, n_pairs)
    count = 1
    for x1 in X1
        for x2 in X2
            test_x_area[:, count] = [x1, x2]
            count += 1
        end
    end
    return test_x_area
end


using Random
function balance_binary_data(data_xy::DataFrame; balancing="undersampling", positive_class_label=1, negative_class_label=2)::DataFrame
    negative_class = data_xy[data_xy[:, end].==negative_class_label, :]
    positive_class = data_xy[data_xy[:, end].==positive_class_label, :]
    size_positive_class = size(positive_class, 1)
    size_negative_class = size(negative_class, 1)
    multiplier = div(size_negative_class, size_positive_class)
    leftover = mod(size_negative_class, size_positive_class)
    if balancing == "undersampling"
        data_xy = vcat(negative_class[1:size(positive_class, 1), :], positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
        leftover_samples = vcat(negative_class[Not(1:maximum_per_class), :], positive_class[Not(1:maximum_per_class), :])
    elseif balancing == "generative"
        new_positive_class = vcat(repeat(positive_class, outer=multiplier - 1), positive_class[1:leftover, :], positive_class)
        data_x = select(new_positive_class, Not([:label]))
        data_y = select(new_positive_class, [:label])
        new_positive_class = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_positive_class = hcat(data_x, data_y)
        data_xy = vcat(negative_class, new_positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
        leftover_samples = nothing
    elseif balancing == "none"
        nothing
    end
    # data_x = Matrix(data_xy)[:, 1:end-1]
    # data_y = data_xy.target
    return data_xy
end

# A handy helper function to normalize our dataset.
function meannormalize(x::Vector{Real}, mean_::Real, std_::Real)
    return (x .- mean_) ./ (std_ .+ convert(eltype(std_), 0.000001))
end

# A handy helper function to normalize our dataset.
function robustscaling(x::Vector{Real}, median_::Real, iqr_::Real)
    return (x .- median_) ./ (iqr_ .+ convert(eltype(std_), 0.000001))
end
# A handy helper function to normalize our dataset.
function minmaxscaling(x::Vector{Real}, max_::Real, min_::Real)
    return (x .- min_) ./ (max_ - min_)
end

function maxabsscaling(x::Vector{Real}, absmax_::Real)
    return x ./ absmax_
end

function pool_test_to_matrix(pool::DataFrame, test::DataFrame, n_input::Int, model_type::String)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Tuple{Array{Float32,2},Array{Float32,2}}}
    pool = Matrix{Float32}(pool)
    test = Matrix{Float32}(test)

    pool_x = pool[:, 1:n_input]
    test_x = test[:, 1:n_input]

    if model_type == "MCMC"
        pool_y = pool[:, end]
        test_y = test[:, end]
    elseif model_type == "XGB"
        pool_y = pool[:, end] .- 1
        test_y = test[:, end] .- 1
    end

    pool = (permutedims(pool_x), permutedims(pool_y))
    test = (permutedims(test_x), permutedims(test_y))
    return pool, test
end

# using EvalMetrics
function performance_stats_binary(ground_truth_, predictions_)
    ground_truth = deepcopy(Int.(vec(ground_truth_)))
    predictions = deepcopy(Int.(vec(predictions_)))
    ground_truth[ground_truth.==2] .= 0
    predictions[predictions.==2] .= 0
    cm = EvalMetrics.ConfusionMatrix(ground_truth, predictions)
    f1 = EvalMetrics.f1_score(cm)
    mcc = EvalMetrics.matthews_correlation_coefficient(cm)
    acc = EvalMetrics.accuracy(cm)
    fpr = EvalMetrics.false_positive_rate(cm)
    # fnr = fnr(cm)
    # tpr = tpr(cm)
    # tnr = tnr(cm)
    prec = EvalMetrics.precision(cm)
    recall = EvalMetrics.true_positive_rate(cm)
    threat_score = EvalMetrics.threat_score(cm)
    return acc, f1, mcc, fpr, prec, recall, threat_score, cm
end

using StatisticalMeasures: rmse, mae
function performance_stats_regression(ground_truth_, predictions_)
    ground_truth = deepcopy(vec(ground_truth_))
    predictions = deepcopy(vec(predictions_))
    mse = rmse(predictions, ground_truth)
    mae = mae(predictions, ground_truth)
    return mse, mae
end

using StatisticalMeasures: recall
function average_accuracy_HM(true_labels, predicted_labels)
	
end

using StatisticalMeasures: f1score, balanced_accuracy, MulticlassTruePositiveRate
using StatsBase: countmap
function performance_stats_multiclass(true_labels, predicted_labels)
	true_labels = vec(true_labels)
	predicted_labels = vec(predicted_labels)
	# missing_labels = setdiff(predicted_labels, true_labels)
	# for missing_label in missing_labels
	# 	true_labels_indices_with_missing_label = findall(!=(missing_label), true_labels)
	# 	predicted_labels_indices_with_missing_label = findall(!=(missing_label), predicted_labels)
	# 	if lastindex(true_labels_indices_with_missing_label) < lastindex(predicted_labels_indices_with_missing_label)
	# 		true_labels = true_labels[true_labels_indices_with_missing_label]
	# 		predicted_labels = predicted_labels[true_labels_indices_with_missing_label]
	# 	else
	# 		true_labels = true_labels[predicted_labels_indices_with_missing_label]
	# 		predicted_labels = predicted_labels[predicted_labels_indices_with_missing_label]
	# 	end
	# end

	writedlm("./test.csv", [true_labels predicted_labels])

	# Convert inputs to categorical vectors
    true_labels = categorical(true_labels, ordered=true)
    predicted_labels = categorical(predicted_labels, ordered=true)
    levels!(predicted_labels, levels(true_labels))

    # Calculate class weights
    # label_dist = sort(countmap(true_labels))
	# pct_labels = collect(values(label_dist)) ./ sum(collect(values(label_dist)))
    # pct_labels = Dict(keys(label_dist) .=> pct_labels)
	# @info "Distribution of Classes" pct_labels

    # n_total = length(true_labels)
    n_classes = lastindex(levels(true_labels))
    # weights = n_total ./ (n_classes .* (collect(values(label_dist))))
    
    # Create a dictionary of class weights
    # class_weights = sort(Dict(keys(label_dist) .=> weights))
	# @info "Class Weights" class_weights
    # Calculate F1 score and accuracy
	if n_classes == 2
		acc = balanced_accuracy(predicted_labels, true_labels)
		f1 = f1score(predicted_labels, true_labels)
	else
		acc = balanced_accuracy(predicted_labels, true_labels)
		mul_recall = MulticlassTruePositiveRate(;average=NoAvg(), levels=levels(true_labels))
		recalls = values(mul_recall(predicted_labels, true_labels))
		f1 = 1 / ((1/n_classes)*(1 ./ recalls))
	end
    @info "Acc and f1" acc f1
    return acc, f1
end


# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    shuffled_indices = shuffle(1:r)
    index = Int(round(r * at))

    pool_indices = shuffled_indices[1:index]
    test_indices = shuffled_indices[(index+1):end]

    pool = df[pool_indices, :]
    test = df[test_indices, :]

    return pool, test
end

function train_validate_test(df; v=0.6, t=0.8)
    r = size(df, 1)
    shuffled_indices = shuffle(1:r)
    val_index = Int(round(r * v))
    test_index = Int(round(r * t))

    train_indices = shuffled_indices[1:val_index]
    validate_indices = shuffled_indices[(val_index+1):test_index]
    test_indices = shuffled_indices[(test_index+1):end]

    train = df[train_indices, :]
    validate = df[validate_indices, :]
    test = df[test_indices, :]

    return train, validate, test
end


function summary_stats(a::AbstractArray{T}) where {T<:Real}
    m = mean(a)
    qs = quantile(a, [0.00, 0.25, 0.50, 0.75, 1.00])
    R = typeof(convert(AbstractFloat, zero(T)))
    stats = Array{R}([convert(R, m),
        convert(R, qs[1]),
        convert(R, qs[2]),
        convert(R, qs[3]),
        convert(R, qs[4]),
        convert(R, qs[5])])
    names_stats = ["Mean",
        "Minimum",
        "1st Quartile",
        "Median",
        "3rd Quartile",
        "Maximum"]
    return [names_stats stats]
end
