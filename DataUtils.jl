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

function pool_test_to_matrix(pool::DataFrame, test::DataFrame, n_input::Int, model_type::String; n_output=nothing)::Tuple{Tuple{Array{Float32,2},Array{Float32,2}},Tuple{Array{Float32,2},Array{Float32,2}}}
    pool = Matrix{Float32}(pool)
    test = Matrix{Float32}(test)

    pool_x = pool[:, 1:n_input]
    test_x = test[:, 1:n_input]

    if model_type == "XGB"
        pool_y = permutedims(pool[:, end] .- 1)
        test_y = permutedims(test[:, end] .- 1)
    else # outputs a tuple = (n_input×total samples Matrix{Float32}, 1×total samples Matrix{Float32})
        pool_y = permutedims(pool[:, end])
        test_y = permutedims(test[:, end])
        # elseif model_type == "Evidential" || model_type == "LaplaceApprox"
        # 	pool_y = Flux.onehotbatch(pool[:, end], 1:n_output)
        #     test_y = Flux.onehotbatch(test[:, end], 1:n_output)
    end

    pool = (permutedims(pool_x), pool_y)
    test = (permutedims(test_x), test_y)
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

using StatisticalMeasures: FScore, balanced_accuracy, MulticlassTruePositiveRate, NoAvg
using StatsBase: countmap
using CategoricalArrays
function performance_stats_multiclass(true_labels, predicted_labels, n_classes)
    # @info "Number of classes" n_classes
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

    # writedlm("./test.csv", [true_labels predicted_labels])

    # Convert inputs to categorical vectors

    true_labels = categorical(true_labels, ordered=true)
    predicted_labels = categorical(predicted_labels, ordered=true)

    levels!(true_labels, 1:n_classes)
    levels!(predicted_labels, 1:n_classes)

    # Calculate class weights
    # label_dist = sort(countmap(true_labels))
    # pct_labels = collect(values(label_dist)) ./ sum(collect(values(label_dist)))
    # pct_labels = Dict(keys(label_dist) .=> pct_labels)
    # @info "Distribution of Classes" pct_labels

    # n_total = length(true_labels)
    # n_classes = lastindex(levels(true_labels))
    # weights = n_total ./ (n_classes .* (collect(values(label_dist))))

    # Create a dictionary of class weights
    # class_weights = sort(Dict(keys(label_dist) .=> weights))
    # @info "Class Weights" class_weights
    # Calculate F1 score and accuracy
    if n_classes == 2
        acc = balanced_accuracy(predicted_labels, true_labels)
        f1score = FScore(; rev=true)
        f1 = f1score(predicted_labels, true_labels)
    else
        acc = balanced_accuracy(predicted_labels, true_labels)
        mul_recall = MulticlassTruePositiveRate(; average=NoAvg(), levels=1:n_classes)
        recalls = values(mul_recall(predicted_labels, true_labels))
        n_classes_true_labels = lastindex(unique(true_labels))
        if n_classes_true_labels < n_classes
            n_classes = n_classes_true_labels
            recalls = filter(!isnan, collect(recalls))
        end
        f1 = 1 / ((1 / n_classes) * sum(recalls .^ -1))
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

using StatsPlots
using Plots: text

function plotter(cr::Matrix{Float32}, cols::Vector{Symbol})::Nothing
    (n, m) = size(cr)
    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="Pearson Correlation Coefficients")
    annotate!([(j, i, text(round(cr[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])
    savefig("./Experiments/pearson_correlations_Uncertainties.pdf")
    return nothing
end

using Statistics
using Distributions

function confidence_interval_95(data)
    # Calculate the sample mean
    mean_val = mean(data)

    # Calculate the sample standard deviation (use `corrected=true` for sample standard deviation)
    std_dev = std(data, corrected=true)

    # Calculate the sample size
    n = length(data)

    # Z-score for 95% confidence interval (for a two-tailed test)
    z = quantile(Normal(), 0.975)

    # Calculate the margin of error
    margin_of_error = z * (std_dev / sqrt(n))

    # # Confidence interval bounds
    # lower_bound = mean_val - margin_of_error
    # upper_bound = mean_val + margin_of_error

    # return (lower_bound, upper_bound)
    return margin_of_error
end

function mean_std_by_variable(group, group_by::Symbol, measurable::Symbol, variable::Symbol, experiment)
    mean_std = DataFrames.combine(groupby(group, variable), measurable => mean, measurable => confidence_interval_95)
    group_name = first(group[!, group_by])
    mean_std[!, group_by] = repeat([group_name], nrow(mean_std))
    CSV.write("./Experiments/$(experiment)/mean_std_$(group_name)_$(measurable)_$(variable).csv", mean_std)
end
function mean_std_by_group(df_folds, group_by::Symbol, variable::Symbol, experiment; list_measurables=[:MSE, :Elapsed, :MAE, :AcceptanceRate, :NumericalErrors])
    for group in groupby(df_folds, group_by)
        for m in list_measurables
            mean_std_by_variable(group, group_by, m, variable, experiment)
        end
    end
end
using Statistics: mean, std

function auc_per_fold(fold::Int, df::DataFrame, group_by::Symbol, measurement::Symbol, experiment::String)
    aucs_acc = []
    list_compared = []
    list_total_training_samples = []
    for i in groupby(df, group_by)
        acc_ = i[!, measurement]
        # n_aocs_samples = ceil(Int, 0.3 * lastindex(acc_))
        n_aocs_samples = lastindex(acc_)
        total_training_samples = last(i[!, :CumulativeTrainedSize])
        println(total_training_samples)
        push!(list_total_training_samples, total_training_samples)
        auc_acc = mean(acc_[1:n_aocs_samples] .- 0.0) / total_training_samples
        push!(list_compared, first(i[!, group_by]))
        append!(aucs_acc, (auc_acc))
    end
    min_total_samples = minimum(list_total_training_samples)
    df = DataFrame(group_by => list_compared, measurement => min_total_samples .* (aucs_acc))
    CSV.write("./Experiments/$(experiment)/auc_$(measurement)_$(fold).csv", df)
end

function auc_mean(n_folds, experiment, group_by::Symbol, measurement::Symbol)
    df = DataFrame()
    for fold = 1:n_folds
        df_ = CSV.read("./Experiments/$(experiment)/auc_$(measurement)_$(fold).csv", DataFrame, header=1)
        df = vcat(df, df_)
    end

    mean_auc = combine(groupby(df, group_by), measurement => mean, measurement => confidence_interval_95)

    CSV.write("./Experiments/$(experiment)/mean_auc_$(measurement).csv", mean_auc)
end

function plotting_measurable_variable(experiment, groupby::Symbol, list_group_names, dataset, variable::Symbol, measurable::Symbol, measurable_mean::Symbol, measurable_std::Symbol, normalised_measurable::Bool)
    width = 6inch
    height = 6inch
    set_default_plot_size(width, height)
    theme = Theme(major_label_font_size=10pt, minor_label_font_size=8pt, key_title_font_size=10pt, key_label_font_size=8pt, key_position=:inside, colorkey_swatch_shape=:circle, key_swatch_size=10pt)
    Gadfly.push_theme(theme)

    df = DataFrame()
    for group_name in list_group_names
        df_ = CSV.read("./Experiments/$(experiment)/mean_std_$(group_name)_$(measurable)_$(variable).csv", DataFrame, header=1)
        # df_[!, :AcquisitonFunction] .= repeat([group_name], nrow(df_))
        df = vcat(df, df_)
    end
    if normalised_measurable
        y_ticks = collect(0:0.1:1.0)
        fig1a = Gadfly.plot(df, x=variable, y=measurable_mean, color=groupby, ymin=df[!, measurable_mean] - df[!, measurable_std], ymax=df[!, measurable_mean] + df[!, measurable_std], Geom.point, Geom.line, Geom.ribbon, Guide.ylabel(String(measurable)), Guide.xlabel(String(variable)), Guide.yticks(ticks=y_ticks), Coord.cartesian(xmin=xmin = df[!, variable][1], ymin=0.0, ymax=1.0))
    else
        fig1a = Gadfly.plot(df, x=variable, y=measurable_mean, color=groupby, ymin=df[!, measurable_mean] - df[!, measurable_std], ymax=df[!, measurable_mean] + df[!, measurable_std], Geom.point, Geom.line, Geom.ribbon, Guide.ylabel(String(measurable)), Guide.xlabel(String(variable)), Coord.cartesian(xmin=xmin = df[!, variable][1]))
    end
    fig1a |> PDF("./Experiments/$(experiment)/$(measurable)_$(variable)_$(dataset)_$(experiment)_folds.pdf", dpi=300)
end


function to_integer_categorical_vector(vec::AbstractVector)
    # Get the unique categories and map them to integers
    unique_vals = unique(vec)
    category_map = Dict(val => i for (i, val) in enumerate(unique_vals))

    # Replace each value in the vector with its corresponding integer
    return [category_map[v] for v in vec]
end

function encode_categorical_data(categorical_data, categories)
    # Create a dictionary mapping each category to an index
    category_to_index = Dict(category => i for (i, category) in enumerate(categories))

    # Encode the categorical data as a list of indices
    categorical_inputs = [category_to_index[category] for category in categorical_data]

    return categorical_inputs
end

function onehotbatch_indices(categorical_inputs::Vector{AbstractVector}, num_categories_list::Vector{Int})
    # Preallocate a vector for the one-hot encoded tensors
    indices_tensor = Vector{Any}(undef, lastindex(categorical_inputs))

    # Loop over the categorical inputs and their corresponding number of categories
    for i in 1:lastindex(categorical_inputs)
        # Apply one-hot encoding for each categorical input
        indices_tensor[i] = Flux.onehotbatch(categorical_inputs[i], 1:num_categories_list[i])
    end

    return indices_tensor
end




function to_binary_vector(vec::AbstractVector)
    return [v == vec[1] ? 1 : 0 for v in vec]
end

# Cross-validation function
function cross_validation_indices(n_samples::Int, n_folds::Int)
    # Ensure valid input
    @assert n_folds > 1 "Number of folds should be greater than 1"
    @assert n_samples >= n_folds "Number of samples should be greater or equal to number of folds"

    # Generate an array of indices
    indices = collect(1:n_samples)

    # Divide the indices into approximately equal folds
    folds = collect(Iterators.partition(indices, minimum([1000, ceil(Int, n_samples / n_folds)])))

    # Prepare the cross-validation splits
    cv_splits = []

    for i in 1:n_folds
        # Train set is the current fold
        train_set = folds[i]

        # Test set is next fold
        val_set = folds[mod1(i + 1, n_folds)]

        # Store the train/validation sets
        push!(cv_splits, (train_set, val_set))
    end

    return cv_splits
end
