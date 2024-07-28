List_TotalCategories = [3]
List_TotalSamples = [10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
using CSV, DelimitedFiles, DataFrames
using StatsPlots
using MLJ
Cases = [1.0, 2.0, 3.0] #:Correct_Maj, :Correct_Min, :Correct_Eq
for case in Cases
    # list_i = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5]
    # list_j = [4, 6, 4, 5, 6, 4, 5, 6, 5, 6, 6]
	list_i = [7,7,7,7,7]
    list_j = [2,3,4,5,6]
    names = [:PctCorrectByMembers, :AvgEntropyCorrect, :AvgEntropyInCorrect, :TotalUncertainty, :Aleatoric, :Epistemic, :VarAleatoric]

    for num_Categories in List_TotalCategories
        plt = plot()
        for (i, j) in zip(list_i, list_j)
            xs = []
            ys = []
            # for (num_Categories, TotalSamples) in zip(List_TotalCategories, List_TotalSamples)
            for TotalSamples in List_TotalSamples
                M = readdlm("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/$(TotalSamples) Samples/pearson_correlations_Uncertainties_NumCategories=$(num_Categories)_TotalSamples=$(TotalSamples)_Case=$(case).csv", ',', Float32)

                push!(ys, M[i, j])
                push!(xs, TotalSamples)
                # push!(xs, num_Categories)
            end
			# LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
			# mach = fit!(machine(LinearRegressor(), DataFrame(x=Vector{Float32}(xs)), Vector{Float32}(ys)))
			# test_x = DataFrame(x=[100])
			# preds = MLJ.predict(mach, test_x)
			# push!(xs, test_x.x[1])
			# push!(ys, preds[1])

            plot!(xs, ys, ylabel="Pearson Correlation", xlabel="Total Number of Samples", ylims=[-1, 1], label="$(names[i]) and $(names[j])", dpi=300, size=(600, 600), xscale=:identity, legend=:outerbottom, linewidth = 2)
        end
        mkpath("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/Pearson Correlation versus TotalSamples VarAleatoric/NumCategories=$(num_Categories)")
        savefig("/Users/456828/Projects/Bayesian-Active-Learning/Simulations of Uncertainty/Pearson Correlation versus TotalSamples VarAleatoric/NumCategories=$(num_Categories)/corr_plot_Case=$(case).png")
    end
end