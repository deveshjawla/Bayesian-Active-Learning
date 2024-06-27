using Random
function data_balancing(data_xy::DataFrame; balancing::String, positive_class_label=1, negative_class_label=2)::DataFrame
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
        data_x = select(new_positive_class, Not([:label]))
        data_y = select(new_positive_class, [:label])
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

function balanced_binary_maker(data_xy::DataFrame; positive_class_label=1, negative_class_label=2, maximum_per_class)
    negative_class = data_xy[data_xy[:, end].==negative_class_label, :]
    positive_class = data_xy[data_xy[:, end].==positive_class_label, :]
    size_positive_class = size(positive_class)[1]
    size_negative_class = size(negative_class)[1]
    if maximum_per_class > minimum([size_negative_class, size_positive_class])
		println(minimum([size_negative_class, size_positive_class]))
        error("maximum_per_class is greater than no of samples in one of the classes")
	else
		data_xy = vcat(negative_class[1:maximum_per_class, :], positive_class[1:maximum_per_class, :])
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
		leftover_samples = vcat(negative_class[Not(1:maximum_per_class), :], positive_class[Not(1:maximum_per_class), :])
    end
    return data_xy, leftover_samples
end

function undersampling(data_xy::Matrix; positive_class_label=1, negative_class_label=2)
    data_xy = data_xy[:, shuffle(axes(data_xy, 2))]
    negative_class = data_xy[:, data_xy[end, :].==negative_class_label]
    positive_class = data_xy[:, data_xy[end, :].==positive_class_label]
    size_positive_class = size(positive_class)[2]
    size_negative_class = size(negative_class)[2]
    if size_negative_class > size_positive_class
        index_max = size_positive_class
		data_xy = hcat(negative_class[:, 1:index_max], positive_class)
		leftover_samples = negative_class[:, index_max+1:end]
	elseif size_negative_class < size_positive_class
        index_max = size_negative_class
		data_xy = hcat(positive_class[:, 1:index_max], negative_class)
		leftover_samples = positive_class[:, index_max+1:end]
	elseif size_negative_class == size_positive_class
		leftover_samples = nothing
    end
    data_xy = data_xy[:, shuffle(axes(data_xy, 2))]
    return data_xy, leftover_samples
end

# A handy helper function to normalize our dataset.
function meannormalize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ convert(eltype(std_), 0.000001))
end

# A handy helper function to normalize our dataset.
function robustscaling(x, median_, iqr_)
    return (x .- median_) ./ (iqr_ .+ convert(eltype(std_), 0.000001))
end
# A handy helper function to normalize our dataset.
function minmaxscaling(x, max_, min_)
    return (x .- min_) ./ (max_ - min_)
end

function maxabsscaling(x, absmax_)
    return x ./ absmax_
end

function pool_test_maker(pool::DataFrame, test::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Tuple{Array{Float32,2},Array{Float32,2}}}
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
    # pool_max = maximum(pool_x, dims=1)
    # pool_mini = minimum(pool_x, dims=1)
    # pool_x = minmaxscaling(pool_x, pool_max, pool_mini)
    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)

    test_x = test[1:n_input, :]
    test_y = test[end, :]
    # test_x = minmaxscaling(test_x, pool_max, pool_mini)
    # test_x = standardize(test_x, pool_mean, pool_std)


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

function pool_test_maker_xgb(pool::DataFrame, test::DataFrame, n_input::Int, n_output::Int)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Tuple{Array{Float32,2},Array{Float32,2}}}
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
	if n_output == 2
    	pool_y = replace(pool[end, :], 2=>0)
	else
		pool_y = pool[end, :] .- 1
	end
    # pool_max = maximum(pool_x, dims=1)
    # pool_mini = minimum(pool_x, dims=1)
    # pool_x = minmaxscaling(pool_x, pool_max, pool_mini)
    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)

    test_x = test[1:n_input, :]
    if n_output == 2
    	test_y = replace(test[end, :], 2=>0)
	else
		test_y = test[end, :] .- 1
	end
    # test_x = minmaxscaling(test_x, pool_max, pool_mini)
    # test_x = standardize(test_x, pool_mean, pool_std)

    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
    df = df[shuffle(axes(df, 1)), :]
    index = Int(round(r * at))
    pool = df[1:index, :]
    test = df[(index+1):end, :]
    return pool, test
end

using EvalMetrics
function performance_stats(ground_truth_, predictions_)
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


function performance_stats_regression(ground_truth_, predictions_)
    ground_truth = deepcopy(vec(ground_truth_))
    predictions = deepcopy(vec(predictions_))  
	mse =  Flux.mse(predictions, ground_truth)
	mae =  Flux.mae(predictions, ground_truth)
    return mse, mae
end

function accuracy_multiclass(true_labels, predictions)
    return mean(true_labels .== predictions)
end

using StatisticalMeasures: macro_f1score, accuracy
using StatsBase: countmap
function performance_stats_multiclass(a,b)
	a = deepcopy(Int.(vec(a)))
    b = deepcopy(Int.(vec(b)))
	weights_ = countmap(a)
	keys_, values_ = keys(weights_), values(weights_)
	n_classes = lastindex(collect(keys(weights_)))
	weights = map(x->lastindex(a)/(n_classes*x), values_)
	class_weights = Dict(keys_ .=> weights)
    f1 = macro_f1score(b,a, class_weights)
    acc = accuracy(b,a, class_weights)
    return acc, f1
end

function train_validate_test(df; v=0.6, t=0.8)
    r = size(df, 1)
    val_index = Int(round(r * v))
    test_index = Int(round(r * t))
    df = df[shuffle(axes(df, 1)), :]
    train = df[1:val_index, :]
    validate = df[(val_index+1):test_index, :]
    test = df[(test_index+1):end, :]
    return train, validate, test
end



# function balanced_accuracy_score(y_pred, y_true)
# 	C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         per_class = np.diag(C) / C.sum(axis=1)
#     if np.any(np.isnan(per_class)):
#         warnings.warn("y_pred contains classes not in y_true")
#         per_class = per_class[~np.isnan(per_class)]
#     score = np.mean(per_class)
#     if adjusted:
#         n_classes = len(per_class)
#         chance = 1 / n_classes
#         score -= chance
#         score /= 1 - chance
#     return score
# end

function summary_stats(a::AbstractArray{T}) where T<:Real
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
