experiments = ["WeightedLikelihood"]

datasets = ["iris1988", "yeast1996"] # "adult1994", "banknote2012", "coalmineseismicbumps", "creditdefault2005", "creditfraud", "iris1988", "stroke", "yeast1996"
minimum_training_sizes = [30, 296] #60, 40, 100, 80, 40, 30, 60, 296
acquisition_sizes = round.(Int, minimum_training_sizes ./ 2)
list_acq_steps = repeat([1], 8)

list_n_folds = [5, 5]#5, 5, 3, 5, 5, 5, 5, 5

list_prior_informativeness = ["UnInformedPrior"] # "UnInformedPrior", "InformedPrior", "NoInit"
list_prior_variance = ["GlorotPrior"] # "GlorotPrior", 0.01, 0.2, 1.0, 3.0, 5.0
list_likelihood_name = ["WeightedLikelihood"] #"UnWeightedLikelihood", "WeightedLikelihood", "Regression"
acq_functions = ["Initial"] # "BayesianUncertainty", "Initial", "Random"
# temperature = nothing, or a Float or list of nothing and Floats, nothing invokes a non-customised Likelihood in the @model
temperatures = [nothing] # 1.0, 0.1, 0.001 or nothing
using DelimitedFiles, DataFrames, CSV, Statistics

col_names = [:Dataset, :Prior, :Likelihood, :OOB_Rhat, :AvgAcceptanceRate, :NumericalErrors, :ESS, :PSRF]
convergence_stats = DataFrame([[] for _ in col_names], col_names)
for (dataset, acquisition_size, n_folds, n_acq_steps) in zip(datasets, acquisition_sizes, list_n_folds, list_acq_steps)
    PATH = @__DIR__
    cd(PATH * "/Data/Tabular/$(dataset)")
    println(dataset)

    ###
    ### Data
    ###

    PATH = pwd()
    stats_df_avg = 0
    stats_df_std = 0
    for prior_informativeness in list_prior_informativeness
        for prior_variance in list_prior_variance
            for likelihood_name in list_likelihood_name
                for acq_func in acq_functions
					for experiment in experiments
                    	for temperature in temperatures
                            stats_df = Array{Any}(undef, 5, n_folds)
                            for fold in 1:n_folds
                                num_mcsteps = 1000
                                num_chains = 2
                                pipeline_name = "./Experiments/$(experiment)/$(acquisition_size)_$(acq_func)_$(prior_variance)_$(likelihood_name)_$(prior_informativeness)_$(temperature)_$(fold)_$(num_chains)_$(num_mcsteps)/convergence_statistics"

                                stats = readdlm("$(pipeline_name)/$(n_acq_steps)_chain.csv")
                                psrf = readdlm("$(pipeline_name)/$(n_acq_steps)_max_psrf.csv")
                                stats_df[:, fold] = vcat(stats[2:5], psrf)
                            end
                            stats_df_avg = mean(stats_df, dims=2)
                            stats_df_std = std(stats_df, dims=2)
                            stats_df_ = []
                            for (i, j) in zip(stats_df_avg, stats_df_std)
                                stats_df_ = vcat(stats_df_, join([string(round(i, digits=2)), " ± ", round(j, digits=2)]))
                            end
                            push!(convergence_stats, vcat(dataset, prior_variance, string(temperature), stats_df_))
                        end
                    end
                end
            end
        end
    end
end

CSV.write("../../../convergence_stats2_1000.csv", convergence_stats)