using CSV, DelimitedFiles, DataFrames
using StatsPlots
using MLJ
List_TotalCategories = [2, 3, 4, 5, 6]
List_TotalSamples = [10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 100, 1000]
Cases = [1.0, 2.0, 3.0] #:Correct_Maj, :Correct_Min, :Correct_Eq
let
    empty_matrix = Array{Any}(undef, 0, 5)
    for case in Cases
        for num_Categories in List_TotalCategories
            for TotalSamples in List_TotalSamples
                M = 0
                try
                    M = readdlm("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/$(TotalSamples) Samples/pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case).csv", ',', Float32)
                catch
                    M = nothing
                end

                if !isnothing(M)
                    list_i = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 7, 7, 7, 7, 7]
                    list_j = [4, 6, 4, 5, 6, 4, 5, 6, 5, 6, 6, 2, 3, 4, 5, 6]
                    names = [:PctCorrectByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]
                    for (i, j) in zip(list_i, list_j)
                        correlation_ = M[i, j]
                        empty_matrix = vcat(empty_matrix, [case num_Categories TotalSamples "$(names[i]) and $(names[j])" correlation_])
                    end
                end
            end
        end
    end
    df = DataFrame(empty_matrix, [:Case, :NumCategories, :TotalSamples, :Pair, :Correlation])
    df = filter(row -> !isnan(row.Correlation), df)
    CSV.write("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/correlations_dataframe.csv", df)
end