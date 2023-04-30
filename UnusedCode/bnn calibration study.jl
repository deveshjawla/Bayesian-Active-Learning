using Distributed
using Turing
# Add four processes to use for sampling.
addprocs(5; exeflags=`--project`)

@everywhere begin
    PATH = @__DIR__
    cd(PATH)

    include("../../BNNUtils.jl")
    include("../../Calibration.jl")
    include("../../DataUtils.jl")

    ###
    ### Data
    ###
    using DataFrames
    using CSV

    train_xy = CSV.read("train.csv", DataFrame, header=1)
    shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
    train_xy = select(train_xy, vcat(shap_importances.feature_name[1:6], "stroke"))
    balanced_data = data_balancing(train_xy, balancing="undersampling")

    using MLJ: partition

    train_xy, validate_xy = partition(balanced_data, 0.8, shuffle=true, rng=1334)

    train_x = Matrix(train_xy[:, 1:end-1])
    # train_max = maximum(train_x, dims=1)
    # train_mini = minimum(train_x, dims=1)
    # train_x = scaling(train_x, train_max, train_mini)
    train_mean = mean(train_x, dims=1)
    train_std = std(train_x, dims=1)
    train_x = standardize(train_x, train_mean, train_std)
    train_y = train_xy[:, end]

    validate_x = Matrix(validate_xy[:, 1:end-1])
    validate_x = standardize(validate_x, train_mean, train_std)
    validate_y = validate_xy[:, end]
    n_validate = lastindex(validate_y)


    test_xy = CSV.read("./test.csv", DataFrame, header=1)
    test_xy = select(test_xy, vcat(shap_importances.feature_name[1:6], "stroke"))
    test_xy = data_balancing(test_xy, balancing="undersampling")
    test_x = Matrix(test_xy[:, 1:end-1])
    # test_x = scaling(test_x, train_max, train_mini)
    test_x = standardize(test_x, train_mean, train_std)
    test_y = test_xy[:, end]
    n_test = lastindex(test_y)

    train_x = Array(train_x')
    validate_x = Array(validate_x')
    test_x = Array(test_x')
    train_y[train_y.==0] .= 2
    validate_y[validate_y.==0] .= 2
    test_y[test_y.==0] .= 2
    name = "calibration_study"
    mkpath("./$(experiment_name)/$(name)")

    ###
    ### Dense Network specifications
    ###

    input_size = size(train_x)[1]
    l1, l2, l3, l4, l5 = 5, 5, 2, 0, 0
    nl1 = input_size * l1 + l1
    nl2 = l1 * l2 + l2
    nl3 = l2 * l3 + l3
    nl4 = l3 * l4 + l4
    ol5 = l4 * l5 + l5

    total_num_params = nl1 + nl2 + nl3 + nl4 + ol5

    using Flux


    # function feedforward(θ::AbstractVector)
    #     W0 = reshape(θ[1:60], 10, 6)
    #     b0 = θ[61:70]
    #     W1 = reshape(θ[71:170], 10, 10)
    #     b1 = θ[171:180]
    #     W2 = reshape(θ[181:280], 10, 10)
    #     b2 = θ[281:290]
    #     W3 = reshape(θ[291:390], 10, 10)
    #     b3 = θ[391:400]
    #     W4 = reshape(θ[401:420], 2, 10)
    #     b4 = θ[421:422]
    #     model = Chain(
    #         Dense(W0, b0, relu),
    #         Dense(W1, b1, relu),
    #         Dense(W2, b2, relu),
    #         Dense(W3, b3, relu),
    #         Dense(W4, b4),
    #         softmax
    #     )
    #     return model
    # end

    function feedforward(θ::AbstractVector)
        W0 = reshape(θ[1:30], 5, 6)
        b0 = reshape(θ[31:35], 5)
        W1 = reshape(θ[36:60], 5, 5)
        b1 = reshape(θ[61:65], 5)
        W2 = reshape(θ[66:75], 2, 5)
        b2 = reshape(θ[76:77], 2)
        model = Chain(
            Dense(W0, b0, relu),
            Dense(W1, b1, relu),
            Dense(W2, b2),
            softmax
        )
        return model
    end


    ###
    ### Bayesian Network specifications
    ###
    using Turing

    # setprogress!(false)
    # using Zygote
    # Turing.setadbackend(:zygote)
    using ReverseDiff
    Turing.setadbackend(:reversediff)

    # sigma = HeGlorot

    #Here we define the layer by layer initialisation
    sigma = vcat(sqrt(2 / (input_size + l1)) * ones(nl1), sqrt(2 / (l1 + l2)) * ones(nl2), sqrt(2 / (l2 + l3)) * ones(nl3), sqrt(2 / (l3 + l4)) * ones(nl4), sqrt(2 / (l4 + l5)) * ones(ol5))
end

# Define a model on all processes.
@everywhere @model bayesnn(x, y) = begin
    θ ~ MvNormal(zeros(total_num_params), ones(total_num_params) .* sigma)
    nn = feedforward(θ)

    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end

@everywhere model = bayesnn(train_x, train_y)
nsteps = 1000
@everywhere n_chains = 5
# Learning with the train set only
chain_timed = @timed sample(model, NUTS(), MCMCDistributed(), nsteps, n_chains)
chain = chain_timed.value
elapsed_ = chain_timed.time
θ = MCMCChains.group(chain, :θ).value
using DelimitedFiles
mkpath("./$(experiment_name)/$(name)/calibration_step/Validate/Uncalibrated")
mkpath("./$(experiment_name)/$(name)/calibration_step/Test/Uncalibrated")
mkpath("./$(experiment_name)/$(name)/calibration_step/Test/Calibrated")
mkpath("./$(experiment_name)/$(name)/post_calibration/Test/Calibrated")
mkpath("./$(experiment_name)/$(name)/post_calibration/Test/Uncalibrated")

using EvalMetrics
# set_encoding(OneTwo())
using Plots

function performance_stats(ground_truth_, predictions_)
	ground_truth = deepcopy(ground_truth_)
    predictions = deepcopy(predictions_)
    ground_truth[ground_truth.==2] .= 0
    predictions[predictions.==2] .= 0
    f1 = f1_score(ground_truth, predictions)
    mcc = matthews_correlation_coefficient(ground_truth, predictions)
    acc = accuracy(ground_truth, predictions)
    fpr = false_positive_rate(ground_truth, predictions)
    # fnr = fnr(ground_truth, predictions)
    # tpr = tpr(ground_truth, predictions)
    # tnr = tnr(ground_truth, predictions)
    prec = precision(ground_truth, predictions)
    recall = true_positive_rate(ground_truth, predictions)
    return acc, mcc, f1, fpr, prec, recall
end

function convergence_stats(i, chain, elapsed)
    ch = chain[:, :, i]
    summaries, quantiles = describe(ch)
    large_rhat = count(>=(1.01), sort(summaries[:, :rhat]))
    small_rhat = count(<=(0.99), sort(summaries[:, :rhat]))
    oob_rhat = large_rhat + small_rhat
    avg_acceptance_rate = mean(ch[:acceptance_rate])
    total_numerical_error = sum(ch[:numerical_error])
    avg_ess = mean(summaries[:, :ess])
    # println(describe(summaries[:, :mean]))
    return elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess
end

function calibration_plot_maker(i, number_of_bins, confidence, ground_truth_, stepname::String, name)
    ground_truth = deepcopy(ground_truth_)
    ground_truth[ground_truth.==2] .= 0
    bins, mean_conf, bin_acc, calibration_gaps = conf_bin_indices(number_of_bins, confidence, ground_truth)

    total_samples = lastindex(confidence)

    ECE, MCE = ece_mce(bins, calibration_gaps, total_samples)

    writedlm("./$(experiment_name)/$(name)/" * stepname * "/ece_mce_$(i).txt", [["ECE", "MCE"] [ECE, MCE]], ',')

    f(x) = x
    reliability_diagram = bar(filter(!isnan, collect(values(mean_conf))), filter(!isnan, collect(values(bin_acc))), legend=false, title="Reliability diagram with \n ECE:$(ECE), MCE:$(MCE)",
        xlabel="Confidence",
        ylabel="# Class labels in Target", size=(800, 600))
    plot!(f, 0, 1, label="Perfect Calibration")
    savefig(reliability_diagram, "./$(experiment_name)/$(name)/" * stepname * "/reliability_diagram_$(i).png")
end

number_of_bins = 3

#before Calibration
for i in 1:n_chains
    params_set = collect.(eachrow(θ[:, :, i]))
    param_matrix = mapreduce(permutedims, vcat, params_set)

    independent_param_matrix = Array{Float64}(undef, Int(nsteps / 10), total_num_params)
    for i in 1:lastindex(param_matrix, 1)
        if i % 10 == 0
            independent_param_matrix[Int((i) / 10), :] = param_matrix[i, :]
        end
    end
    writedlm("./$(experiment_name)/$(name)/calibration_step/param_matrix_$(i).csv", independent_param_matrix, ',')

    ŷ_validate, pŷ_validate = pred_analyzer_multiclass(validate_x, validate_y, independent_param_matrix)
    ŷ_test, pŷ_test = pred_analyzer_multiclass(test_x, independent_param_matrix)

    # println(countmap(ŷ_test))

    writedlm("./$(experiment_name)/$(name)/calibration_step/ŷ_validate_$(i).csv", hcat(ŷ_validate, pŷ_validate), ',')
    writedlm("./$(experiment_name)/$(name)/calibration_step/ŷ_test_$(i).csv", hcat(ŷ_test, pŷ_test), ',')

    # gr()
    # prplot(test_y, pŷ_test)
    # no_skill(x) = count(==(1), test_y) / length(test_y)
    # plot!(no_skill, 0, 1, label="No Skill Classifier")
    # savefig("./$(experiment_name)/$(name)/PRCurve_$(i).png")
    # prauc = au_prcurve(test_y, pŷ_test)


    elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chain, elapsed_)

    writedlm("./$(experiment_name)/$(name)/calibration_step/convergence_statistics_chain_$(i).csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')


    acc, mcc, f1, fpr, prec, recall = performance_stats(validate_y, ŷ_validate)
    writedlm("./$(experiment_name)/$(name)/calibration_step/performance_statistics_validate_chain_$(i).csv", [["Accuracy", "MCC", "f1", "fpr", "precision", "recall"] [acc, mcc, f1, fpr, prec, recall]], ',')

    acc, mcc, f1, fpr, prec, recall = performance_stats(test_y, ŷ_test)
    writedlm("./$(experiment_name)/$(name)/calibration_step/performance_statistics_test_chain_$(i).csv", [["Accuracy", "MCC", "f1", "fpr", "precision", "recall"] [acc, mcc, f1, fpr, prec, recall]], ',')

    calibration_plot_maker(i, number_of_bins, pŷ_validate, validate_y, "calibration_step/Validate/Uncalibrated", name)
    calibration_plot_maker(i, number_of_bins, pŷ_test, test_y, "calibration_step/Test/Uncalibrated", name)
	println(pŷ_validate[1:5])
    loss((a, b)) = _loss_binary(a, b, pŷ_validate, validate_y)
    @time result = optimize(loss, [1.0, 1.0], LBFGS())
    a, b = result.minimizer
    calibrated_pŷ_test = platt(pŷ_test .* a .+ b)
	println(calibrated_pŷ_test[1:5])
    writedlm("./$(experiment_name)/$(name)/calibration_step/calibration_fit_params.csv", [a, b], ',')

    calibration_plot_maker(i, number_of_bins, calibrated_pŷ_test, test_y, "calibration_step/Test/Calibrated", name)
end

# sum([idx * i for (i, idx) in enumerate(summaries[:, :mean])])
# _, i = findmax(chain[:lp])
# i = i.I[1]
# elapsed = chain_timed.time
# θ = MCMCChains.group(chain, :θ).value
# θ[i, :]
begin
    avg_convergence_stats = Array{Any}(undef, 5, n_chains)
    avg_perf_stats_validate = Array{Any}(undef, 6, n_chains)
    avg_perf_stats_test = Array{Any}(undef, 6, n_chains)
    avg_ece_mce_validate = Array{Any}(undef, 2, n_chains)
    avg_ece_mce_test_uncalibrated = Array{Any}(undef, 2, n_chains)
    avg_ece_mce_test_calibrated = Array{Any}(undef, 2, n_chains)
    for i = 1:n_chains
        a = readdlm("./$(experiment_name)/$(name)/calibration_step/convergence_statistics_chain_$(i).csv", ',')
        b = readdlm("./$(experiment_name)/$(name)/calibration_step/performance_statistics_validate_chain_$(i).csv", ',')
        c = readdlm("./$(experiment_name)/$(name)/calibration_step/performance_statistics_test_chain_$(i).csv", ',')
        avg_convergence_stats[:, i] = a[:, 2]
        avg_perf_stats_validate[:, i] = b[:, 2]
        avg_perf_stats_test[:, i] = c[:, 2]
        e = readdlm("./$(experiment_name)/$(name)/calibration_step/Validate/Uncalibrated/ece_mce_$(i).txt", ',')
        f = readdlm("./$(experiment_name)/$(name)/calibration_step/Test/Uncalibrated/ece_mce_$(i).txt", ',')
        g = readdlm("./$(experiment_name)/$(name)/calibration_step/Test/Calibrated/ece_mce_$(i).txt", ',')
        avg_ece_mce_validate[:, i] = e[:, 2]
        avg_ece_mce_test_uncalibrated[:, i] = f[:, 2]
        avg_ece_mce_test_calibrated[:, i] = g[:, 2]
    end
    _a = mean(avg_convergence_stats, dims=2)
    _b = mean(avg_perf_stats_validate, dims=2)
    _c = mean(avg_perf_stats_test, dims=2)
    _e = mean(avg_ece_mce_validate, dims=2)
    _f = mean(avg_ece_mce_test_uncalibrated, dims=2)
    _g = mean(avg_ece_mce_test_calibrated, dims=2)
    mkpath("./$(experiment_name)/$(name)/calibration_step/means")
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/convergence_statistics.txt", _a)
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/performance_statistics_validate.txt", _b)
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/performance_statistics_test.txt", _c)
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/ece_mce_validate.txt", _e)
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/ece_mce_test_uncalibrated.txt", _f)
    writedlm("./$(experiment_name)/$(name)/calibration_step/means/ece_mce_test_calibrated.txt", _g)
end

begin
    param_matrices_accumulated = Array{Float64}(undef, n_chains*Int(nsteps / 10), total_num_params)
    for i in 1:n_chains
        a = readdlm("./$(experiment_name)/$(name)/calibration_step/param_matrix_$(i).csv", ',')
        param_matrices_accumulated[(i-1)*100+1:i*100,:] = a
    end
    param_matrix_mean = mean(param_matrices_accumulated, dims=1)
    param_matrix_std = std(param_matrices_accumulated, dims=1)
    println("The mean param operation correct?   ", lastindex(param_matrix_mean) == total_num_params)
end

param_matrix_mean = vec(param_matrix_mean)
param_matrix_std = vec(param_matrix_std)
@everywhere param_matrix_mean = $param_matrix_mean
@everywhere param_matrix_std = $param_matrix_std

@everywhere @model bayesnn(x, y) = begin
    θ ~ MvNormal(param_matrix_mean, param_matrix_std)
    nn = feedforward(θ)

    ŷ = nn(x)
    for i = 1:lastindex(y)
        y[i] ~ Categorical(ŷ[:, i])
    end
end

@everywhere model_new = bayesnn(validate_x, validate_y)
nsteps = 1000
n_chains = 5
# Learning with the train set only
chain_timed = @timed sample(model_new, NUTS(), MCMCDistributed(), nsteps, n_chains)
chain = chain_timed.value
elapsed_ = chain_timed.time
θ = MCMCChains.group(chain, :θ).value

#After Calibration
for i in 1:n_chains
    params_set = collect.(eachrow(θ[:, :, i]))
    param_matrix = mapreduce(permutedims, vcat, params_set)

    independent_param_matrix = Array{Float64}(undef, Int(nsteps / 10), total_num_params)
    for i in 1:lastindex(param_matrix, 1)
        if i % 10 == 0
            independent_param_matrix[Int((i) / 10), :] = param_matrix[i, :]
        end
    end
    writedlm("./$(experiment_name)/$(name)/post_calibration/param_matrix_$(i).csv", independent_param_matrix, ',')

    ŷ_test, pŷ_test = pred_analyzer_multiclass(test_x, independent_param_matrix)

    # println(countmap(ŷ_test))

    writedlm("./$(experiment_name)/$(name)/post_calibration/ŷ_test_$(i).csv", hcat(ŷ_test, pŷ_test), ',')

    # gr()
    # prplot(test_y, pŷ_test)
    # no_skill(x) = count(==(1), test_y) / length(test_y)
    # plot!(no_skill, 0, 1, label="No Skill Classifier")
    # savefig("./$(experiment_name)/$(name)/PRCurve_$(i).png")
    # prauc = au_prcurve(test_y, pŷ_test)


    elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess = convergence_stats(i, chain, elapsed_)

    writedlm("./$(experiment_name)/$(name)/post_calibration/convergence_statistics_chain_$(i).csv", [["elapsed", "oob_rhat", "avg_acceptance_rate", "total_numerical_error", "avg_ess"] [elapsed, oob_rhat, avg_acceptance_rate, total_numerical_error, avg_ess]], ',')

    acc, mcc, f1, fpr, prec, recall = performance_stats(test_y, ŷ_test)
    writedlm("./$(experiment_name)/$(name)/post_calibration/performance_statistics_test_chain_$(i).csv", [["Accuracy", "MCC", "f1", "fpr", "precision", "recall"] [acc, mcc, f1, fpr, prec, recall]], ',')


    calibration_plot_maker(i, number_of_bins, pŷ_test, test_y, "post_calibration/Test/Uncalibrated", name)

    a, b = readdlm("./$(experiment_name)/$(name)/calibration_step/calibration_fit_params.csv", ',')
	println(pŷ_test[10:15])
    calibrated_pŷ_test = platt(pŷ_test .* a .+ b)
	println(calibrated_pŷ_test[10:15])

    calibration_plot_maker(i, number_of_bins, calibrated_pŷ_test, test_y, "post_calibration/Test/Calibrated", name)
end

begin
    avg_convergence_stats = Array{Any}(undef, 5, n_chains)
    avg_perf_stats_test = Array{Any}(undef, 6, n_chains)
    avg_ece_mce_test_uncalibrated = Array{Any}(undef, 2, n_chains)
    avg_ece_mce_test_calibrated = Array{Any}(undef, 2, n_chains)
    for i = 1:n_chains
        a = readdlm("./$(experiment_name)/$(name)/post_calibration/convergence_statistics_chain_$(i).csv", ',')
        c = readdlm("./$(experiment_name)/$(name)/post_calibration/performance_statistics_test_chain_$(i).csv", ',')
        avg_convergence_stats[:, i] = a[:, 2]
        avg_perf_stats_test[:, i] = c[:, 2]
        f = readdlm("./$(experiment_name)/$(name)/post_calibration/Test/Uncalibrated/ece_mce_$(i).txt", ',')
        g = readdlm("./$(experiment_name)/$(name)/post_calibration/Test/Calibrated/ece_mce_$(i).txt", ',')
        avg_ece_mce_test_uncalibrated[:, i] = f[:, 2]
        avg_ece_mce_test_calibrated[:, i] = g[:, 2]
    end
    _a = mean(avg_convergence_stats, dims=2)
    _c = mean(avg_perf_stats_test, dims=2)
    _f = mean(avg_ece_mce_test_uncalibrated, dims=2)
    _g = mean(avg_ece_mce_test_calibrated, dims=2)
	mkpath("./$(experiment_name)/$(name)/post_calibration/means")
    writedlm("./$(experiment_name)/$(name)/post_calibration/means/convergence_statistics.txt", _a)
    writedlm("./$(experiment_name)/$(name)/post_calibration/means/performance_statistics_test.txt", _c)
    writedlm("./$(experiment_name)/$(name)/post_calibration/means/ece_mce_test_uncalibrated.txt", _f)
    writedlm("./$(experiment_name)/$(name)/post_calibration/means/ece_mce_test_calibrated.txt", _g)
end

