using Statistics, DataFrames
using StaticArrays
using CSV, DataFrames, ProgressMeter
using StatsPlots
using DelimitedFiles

function fast_var_aleatoric(AvgEntropyCorrect::Float16, AvgEntropyInCorrect::Float16, PctCorrectByMembers::SMatrix, TotalSamples::Int)::Float16
    len_correct = round(Int, PctCorrectByMembers[1] * TotalSamples)
    len_incorrect = round(Int, PctCorrectByMembers[2] * TotalSamples)
    total_len = len_correct + len_incorrect

    # Preallocate the array
    concatenated_array = Vector{Float16}(undef, total_len)

    # Fill the array with the correct values
    concatenated_array[1:len_correct] .= AvgEntropyCorrect
    concatenated_array[len_correct+1:end] .= AvgEntropyInCorrect

    # Compute the standard deviation
    std_ = std(concatenated_array)
    return std_
end

function normalized_entropy(prob_vec::SVector, n_output::Int64)::Float16
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum(prob_vec) < 0.99
        return error("sum(prob_vec) is not 1 BUT $(sum(prob_vec)) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log.(prob_vec))) / log(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::SMatrix, n_output::Int64)::Float16
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    return H
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

The H is the Total Uncertainty
E_H is the Aleatoric
H - E_H is the Epistemic

Returns : H (macro_entropy), E_H (mean of Entropies) is the Aleatoric, H - E_H(Epistemic unceratinty)
"""
function bald(prob_matrix::SMatrix, n_output::Int64)::Tuple{Float16,Float16,Float16}
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H - E_H
end

"""
Returns the index of the maximum value in a Vector, if there are no Maximums or Multiple Maximums then nothing is returned
"""
function only_one_argmax(A::Vector{<:Number})::Union{Int64,Nothing}
    _index = findall(x -> x == maximum(A), A)
    if lastindex(_index) == 1
        return first(_index)
    else
        return nothing
    end
end

function only_one_argmin(A::Vector{<:Number})::Union{Int64,Nothing}
    _index = findall(x -> x == minimum(A), A)
    if lastindex(_index) == 1
        return first(_index)
    else
        return nothing
    end
end

function neither_argmax_nor_argmin(A::Vector{<:Number})::Union{Vector{Int64},Nothing}
    _index = findall(x -> x == maximum(A), A)
    if lastindex(_index) > 1
        return _index
    else
        return nothing
    end
end

function analyse_uncertainties(TotalSamples::Int64, num_Categories::Int64, case::Int)::Tuple{Matrix{Float16},Int64}
    if num_Categories == 2
        Possible_Ratio_Pred_CorrectInCorrect = [[i j] for i in 0:TotalSamples, j in 0:TotalSamples if sum([i, j]) == TotalSamples]
    elseif num_Categories == 3
        # Possible_Ratio_Pred_CorrectInCorrect = [[i j k] for i in 0:TotalSamples, j in 0:TotalSamples, k in 0:TotalSamples if sum([i, j, k]) == TotalSamples]
        Possible_Ratio_Pred_CorrectInCorrect = Vector{Matrix{Int}}()
        for i in 0:TotalSamples
            for j in 0:TotalSamples
                k = TotalSamples - i - j
                if k >= 0
                    push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k])
                end
            end
        end
    elseif num_Categories == 4
        # Possible_Ratio_Pred_CorrectInCorrect = [[i j k l] for i in 0:TotalSamples, j in 0:TotalSamples, k in 0:TotalSamples, l in 0:TotalSamples if sum([i, j, k, l]) == TotalSamples]
        Possible_Ratio_Pred_CorrectInCorrect = Vector{Matrix{Int}}()
        for i in 0:TotalSamples
            for j in 0:TotalSamples
                for k in 0:TotalSamples
                    l = TotalSamples - i - j - k
                    if l >= 0
                        push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k l])
                    end
                end
            end
        end
    elseif num_Categories == 5
        Possible_Ratio_Pred_CorrectInCorrect = Vector{Matrix{Int}}()
        for i in 0:TotalSamples
            for j in 0:TotalSamples
                for k in 0:TotalSamples
                    for l in 0:TotalSamples
                        m = TotalSamples - i - j - k - l
                        if m >= 0
                            push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k l m])
                        end
                    end
                end
            end
        end
    elseif num_Categories == 6
        Possible_Ratio_Pred_CorrectInCorrect = Vector{Matrix{Int}}()
        for i in 0:TotalSamples
            for j in 0:TotalSamples
                for k in 0:TotalSamples
                    for l in 0:TotalSamples
                        for m in 0:TotalSamples
                            n = TotalSamples - i - j - k - l - m
                            if n >= 0
                                push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k l m n])
                            end
                        end
                    end
                end
            end
        end
    end

    C_Maj = convert(Vector{SMatrix{1,num_Categories,Float16,num_Categories}}, filter(x -> only_one_argmax(vec(x)) == 1, Possible_Ratio_Pred_CorrectInCorrect) ./ TotalSamples) #Majority of the ensemble mebers vote as correct
    C_Min = convert(Vector{SMatrix{1,num_Categories,Float16,num_Categories}}, filter(x -> only_one_argmax(vec(x)) != 1 && only_one_argmax(vec(x)) !== nothing, Possible_Ratio_Pred_CorrectInCorrect) ./ TotalSamples)
    C_Eq = convert(Vector{SMatrix{1,num_Categories,Float16,num_Categories}}, filter(x -> !isnothing(neither_argmax_nor_argmin(vec(x))), Possible_Ratio_Pred_CorrectInCorrect) ./ TotalSamples)

    List_AverageSoftmaxofCorrect = deepcopy(C_Maj)
    List_AverageSoftmaxofInCorrect = deepcopy(C_Min)

    # total_cases = (lastindex(C_Maj, 1) * lastindex(List_AverageSoftmaxofCorrect, 1) * lastindex(List_AverageSoftmaxofInCorrect, 1)) + (lastindex(C_Min, 1) * lastindex(List_AverageSoftmaxofCorrect, 1) * lastindex(List_AverageSoftmaxofInCorrect, 1)) + (lastindex(C_Eq, 1) * lastindex(List_AverageSoftmaxofCorrect, 1) * lastindex(List_AverageSoftmaxofInCorrect, 1))

    # list_ = Tuple{Int64, SMatrix{1, num_Categories, Float16, num_Categories}, SMatrix{1, num_Categories, Float16, num_Categories}, SMatrix{1, num_Categories, Float16, num_Categories}}[]
    # list_ = []
    # @time for case in Cases
    #     Ratio = 0
    #     if case == 1
    #         Ratio = C_Maj
    #     elseif case == 2
    #         Ratio = C_Min
    #     elseif case == 3
    #         Ratio = C_Eq
    #     end
    #     for PctCorrectByMembers in Ratio, AverageSoftmaxofCorrect in List_AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect in List_AverageSoftmaxofInCorrect
    #         push!(list_, (case, PctCorrectByMembers, AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect))
    #     end
    # end

    # Assuming C_Maj, C_Min, and C_Eq are predefined arrays or values

    if case == 1
        Ratio = C_Maj
    elseif case == 2
        Ratio = C_Min
    elseif case == 3
        Ratio = C_Eq
    end

    total_cases = (lastindex(Ratio, 1) * lastindex(List_AverageSoftmaxofCorrect, 1) * lastindex(List_AverageSoftmaxofInCorrect, 1))
    @info "Total Samples are the total number of cases = $(total_cases)"


    # Ratios = Dict(1 => C_Maj, 2 => C_Min, 3 => C_Eq)
    list_ = Vector{Tuple{SMatrix{1,num_Categories,Float16,num_Categories},SMatrix{1,num_Categories,Float16,num_Categories},SMatrix{1,num_Categories,Float16,num_Categories}}}(undef, total_cases)
    index = 1
    # for case in Cases
    # Ratio = Ratios[case]
    for PctCorrectByMembers in Ratio
        for AverageSoftmaxofCorrect in List_AverageSoftmaxofCorrect
            for AverageSoftmaxofInCorrect in List_AverageSoftmaxofInCorrect
                list_[index] = (PctCorrectByMembers, AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect)
                index += 1
            end
        end
    end
    # end
    # array_size_MB = Base.summarysize(list_)*1e-6

    empty_matrix = Matrix{Float16}(undef, total_cases, 7)
    # array_size_GB = Base.summarysize(empty_matrix)*1e-9

    @showprogress Threads.@threads for i = 1:total_cases
        PctCorrectByMembers, AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect = list_[i]
        PctCorrectByMembers = SMatrix{1,2,Float16,2}(PctCorrectByMembers[1], 1 - PctCorrectByMembers[1])
        AverageSoftmax = PctCorrectByMembers * vcat(AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect)
        TotalUncertainty = normalized_entropy(vec(AverageSoftmax), num_Categories)
        AvgEntropyCorrect = normalized_entropy(vec(AverageSoftmaxofCorrect), num_Categories)
        AvgEntropyInCorrect = normalized_entropy(vec(AverageSoftmaxofInCorrect), num_Categories)
        Aleatoric = PctCorrectByMembers * vcat(AvgEntropyCorrect, AvgEntropyInCorrect)
        VarAleatoric = fast_var_aleatoric(AvgEntropyCorrect, AvgEntropyInCorrect, PctCorrectByMembers, TotalSamples)
        Epistemic = TotalUncertainty - first(Aleatoric)
        empty_matrix[i, :] = [first(PctCorrectByMembers) AvgEntropyCorrect AvgEntropyInCorrect TotalUncertainty first(Aleatoric) Epistemic VarAleatoric]
    end
    return empty_matrix, total_cases
end

function plotter(cr::Matrix{Float32}, num_Categories::Int, TotalSamples::Int, case, cols::Vector)::Nothing
    (n, m) = size(cr)
    heatmap([i > j ? NaN : cr[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=300, size=(800, 700), title="NumCategories=$(num_Categories) TotalSamples=$(TotalSamples) Case=$(case)")
    annotate!([(j, i, text(round(cr[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])
    savefig("./Simulations of Uncertainty/$(TotalSamples) Samples/pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case).pdf")
    return nothing
end

##================================================================

##================================================================

List_TotalCategories = [3, 4]
List_TotalSamples = [100]
using DelimitedFiles
Cases = [1, 2, 3] #:Correct_Maj, :Correct_Min, :Correct_Eq
for num_Categories in List_TotalCategories
    for TotalSamples in List_TotalSamples
        for case in Cases
            @info "Now running simulations for $(num_Categories) Categories! and Case = $(case)"
            results_timed = @timed analyse_uncertainties(TotalSamples, num_Categories, case)
            elapsed = results_timed.time
            M, total_permutations = results_timed.value
            # writedlm("./Simulations of Uncertainty/$(TotalSamples) Samples/unceratinties_theory_data_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case)_TotalPermutations=$(total_permutations)_Elapsed=$(elapsed)seconds.csv", M, ',')
            # results = readdlm("./unceratinties_theory_data_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case)_TotalPermutations=$(total_permutations)_Elapsed=$(elapsed)seconds.csv", ',', Float32)
            mkpath("./Simulations of Uncertainty/$(TotalSamples) Samples")
            M = convert(Matrix{Float32}, M)
            cols = [:PctCorrectByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]
            M = cor(M)
            writedlm("./Simulations of Uncertainty/$(TotalSamples) Samples/pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(Float16(case)).csv", M, ',')
            # M=readdlm("./pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case).csv", ',', Float32)
            plotter(M, num_Categories, TotalSamples, case, cols)
        end
    end
end