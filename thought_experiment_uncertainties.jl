using Statistics, DataFrames
include("./ScoringFunctions.jl")
using CSV, DataFrames, ProgressMeter
using Plots

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

function analyse_uncertainties(TotalSamples::Int64, num_Categories::Int64)::Array{Any,2}
    Possible_Ratio_Pred_CorrectInCorrect = Matrix{Int64}[]
    Possible_Softmax_Vectors = Matrix{Int64}[]
    if num_Categories == 2
        ### 2 Categories
        for i = 0:TotalSamples, j = 0:TotalSamples
            if sum([i j]) == TotalSamples
                push!(Possible_Ratio_Pred_CorrectInCorrect, [i j])
            end
        end
        for i = 0:TotalSamples, j = 0:TotalSamples
            if sum([i j]) == TotalSamples
                push!(Possible_Softmax_Vectors, [i j])
            end
        end

    elseif num_Categories == 3
        for i = 0:TotalSamples, j = 0:TotalSamples, k = 0:TotalSamples
            if sum([i j k]) == TotalSamples
                push!(Possible_Softmax_Vectors, [i j k])
            end
        end
        for i = 0:TotalSamples, j = 0:TotalSamples, k = 0:TotalSamples
            if sum([i j k]) == TotalSamples
                push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k])
            end
        end
    elseif num_Categories == 4
        for i = 0:TotalSamples, j = 0:TotalSamples, k = 0:TotalSamples, l = 0:TotalSamples
            if sum([i j k l]) == TotalSamples
                push!(Possible_Softmax_Vectors, [i j k l])
            end
        end
        for i = 0:TotalSamples, j = 0:TotalSamples, k = 0:TotalSamples, l = 0:TotalSamples
            if sum([i j k l]) == TotalSamples
                push!(Possible_Ratio_Pred_CorrectInCorrect, [i j k l])
            end
        end
    end


    C_Maj = Possible_Ratio_Pred_CorrectInCorrect[BitVector([only_one_argmax(vec(x)) == 1 for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples #Majority of the ensemble mebers vote as correct
    C_Min = Possible_Ratio_Pred_CorrectInCorrect[BitVector([only_one_argmax(vec(x)) != 1 && only_one_argmax(vec(x)) !== nothing for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples
    C_Eq = Possible_Ratio_Pred_CorrectInCorrect[BitVector([!isnothing(neither_argmax_nor_argmin(vec(x))) for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples
    Cases = [:Correct_Maj, :Correct_Min, :Correct_Eq]

    Softmax_of_Correct = Possible_Softmax_Vectors[BitVector([only_one_argmax(vec(x)) == 1 for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples
    Softmax_of_InCorrect = Possible_Softmax_Vectors[BitVector([only_one_argmax(vec(x)) != 1 && only_one_argmax(vec(x)) !== nothing for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples
    Softmax_of_Nil = Possible_Softmax_Vectors[BitVector([!isnothing(neither_argmax_nor_argmin(vec(x))) for x in Possible_Ratio_Pred_CorrectInCorrect])] ./ TotalSamples

    empty_matrix = Array{Any}(undef, 0, 8)

    for case in Cases
        Ratio = 0
        if case == :Correct_Maj
            Ratio = C_Maj
        elseif case == :Correct_Min
            Ratio = C_Min
        elseif case == :Correct_Eq
            Ratio = C_Eq
        end

        @showprogress "Computing Uncertainties for Case $(case) ..." for PercentCorrectPredsByMembers in Ratio, AverageSoftmaxofCorrect in Softmax_of_Correct, AverageSoftmaxofInCorrect in Softmax_of_InCorrect
            PercentCorrectPredsByMembers = [PercentCorrectPredsByMembers[1] 1 - PercentCorrectPredsByMembers[1]]
			AverageSoftmax = PercentCorrectPredsByMembers * vcat(AverageSoftmaxofCorrect, AverageSoftmaxofInCorrect)
            TotalUncertainty = normalized_entropy(vec(AverageSoftmax), num_Categories)
            AvgEntropyCorrect = normalized_entropy(vec(AverageSoftmaxofCorrect), num_Categories)
            AvgEntropyInCorrect = normalized_entropy(vec(AverageSoftmaxofInCorrect), num_Categories)
            Aleatoric = PercentCorrectPredsByMembers * vcat(AvgEntropyCorrect, AvgEntropyInCorrect)
            VarAleatoric = std(vcat(repeat([AvgEntropyCorrect], round(Int, PercentCorrectPredsByMembers[1] * TotalSamples)), repeat([AvgEntropyInCorrect], round(Int, PercentCorrectPredsByMembers[2] * TotalSamples))))
            Epistemic = TotalUncertainty - first(Aleatoric)
            empty_matrix = vcat(empty_matrix, [case first(PercentCorrectPredsByMembers) AvgEntropyCorrect AvgEntropyInCorrect TotalUncertainty first(Aleatoric) Epistemic VarAleatoric])
        end
    end

    return empty_matrix
end

function plotting_correlation_matrices(TotalSamples::Int64, num_Categories::Int64, results::DataFrame)::Nothing
    for df in groupby(results, :Case)
        cols = [:PercentCorrectPredsByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]  # define subset
        M = cor(Matrix(df[!, cols]))
        (n, m) = size(M)
        heatmap([i > j ? NaN : M[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=600, size=(800, 700), title="NumCategories=$(num_Categories) TotalSamples=$(TotalSamples) Case=$(first(df).Case)")
        annotate!([(j, i, text(round(M[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])
        savefig("./pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(first(df).Case).png")
    end

    cols = [:PercentCorrectPredsByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]  # define subset
    M = cor(Matrix(results[!, cols]))
    (n, m) = size(M)
    heatmap([i > j ? NaN : M[i, j] for i in 1:m, j in 1:n], fc=cgrad([:red, :white, :dodgerblue4]), clim=(-1.0, 1.0), xticks=(1:m, cols), xrot=90, yticks=(1:m, cols), yflip=true, dpi=600, size=(800, 700), title="NumCategories=$(num_Categories) TotalSamples=$(TotalSamples) Case=All")
    annotate!([(j, i, text(round(M[i, j], digits=3), 10, "Computer Modern", :black)) for i in 1:n for j in 1:m])
    savefig("./pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=All.png")
	return nothing
end


##================================================================

##================================================================

List_TotalCategories = [2, 3, 4]
TotalSamples = 100

for n_cat in List_TotalCategories
	@info "Now running simulations for $(n_cat) Categories!"
    results_timed = @timed analyse_uncertainties(TotalSamples, n_cat)
	elapsed= results_timed.time
    names = [:Case, :PercentCorrectPredsByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]
    results = DataFrame(results_timed.value, names)
    CSV.write("./unceratinties_theory_data_NumCategories=$(n_cat)_TotalSamples=$(TotalSamples)_Elapsed_$(elapsed)seconds.csv", results)

    plotting_correlation_matrices(TotalSamples,  n_cat, results)
end