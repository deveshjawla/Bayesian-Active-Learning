using Random
function random_acquisition(pool_size::Int, acquisition_size::Int)
    return randperm(pool_size)[1:acquisition_size]
end


using DataFrames
"""
top_k_acquisition_no_duplicates returns a pool of samples according to the following prescription:
0. pool_scores = takes as input the scores of all the samples in the pool
1. acquisition_size = the number of samples to be selected for active learning
2. descending = default False, when True it uses a descending order to sort the pool scores
3. remove_zeros = default False, when True it removes the samples with a score of zero
4. sensitivity = in order to avoid acquiring duplicate samples, we bin the samples according to the scores, those samples which have the same score or scores within the sensitivity interval are put together in a bin and then only one sample from each bin is sampled
"""
function top_k_acquisition_no_duplicates(pool_scores::Vector, acquisition_size::Int; descending=false, remove_zeros=false, sensitivity=0.0001)
    df = DataFrame(Scores=pool_scores, Sample_indices=collect(1:lastindex(pool_scores)))

    #Binning so that similar samples are sampled only once, one sample per bin
    max_score = maximum(df.Scores)
    min_score = minimum(df.Scores)
    bins = min_score:sensitivity:max_score
    # Creating categorical bins
    df.scores_bins = cut(df.Scores, bins, extend=true)
    cut_df = combine(groupby(df, :scores_bins), first)

    # Sorting the DataFrame based on Scores
    sorted_df = sort(cut_df, :Scores, rev=descending)
    # CSV.write("./top_k_acquisition_$(descending).csv", sorted_df)
    # Handling zero scores if required
    if remove_zeros
        non_zero_scores = filter(row -> row.Scores > sensitivity, sorted_df)
        top_k = non_zero_scores[1:min(acquisition_size, nrow(non_zero_scores)), :Sample_indices]
    else
        top_k = sorted_df[1:min(acquisition_size, nrow(sorted_df)), :Sample_indices]
    end
    return top_k
end

using Distributions
using CategoricalArrays
"""
Acquisition Function based on the paper "Stochastic BALD (2022)"
Uses a Gumbel Distribution to add noise to BALD or Softmax scores.

Optional Arguements:
1. acquisition_type = "Power" or "Stochastic" for the Power and Stoachstic Versions of the paper
2. descending = by default is True, sorts the scores in the descending order
3. β = parameter for the Gumbel Distribution (see "Stochastic BALD (2022)" paper for more details)
"""
function stochastic_acquisition(pool_scores::Vector, acquisition_size::Int; acquisition_type="Stochastic", β=1.0, descending=true)
    gumbel_dist = Gumbel(0, β^-1)
    # n = lastindex(pool_scores)
    # gumbel_noise = -log.(-log.(rand(n))) * β #faster version of Gumbel Distribution
    if acquisition_type == "Power"
        scores = log.(pool_scores) .+ rand(gumbel_dist, lastindex(pool_scores))
    elseif acquisition_type == "Stochastic"
        scores = pool_scores .+ rand(gumbel_dist, lastindex(pool_scores))
    end
    indices = partialsortperm(scores, 1:acquisition_size, rev=descending)
    return indices
end

function get_sampled_indices(al_sampling, acq_size_, pool_size, pool_prediction_matrix; reconstruct=nothing, pct_epistemic_samples = 0.8)::Vector{Int}
    if al_sampling == "Initial"
        sampled_indices = 1:acq_size_
    elseif al_sampling == "Random"
        sampled_indices = random_acquisition(pool_size, acq_size_)
    elseif al_sampling == "PowerBALD"
        bald_scores = pool_prediction_matrix[5, :]
        sampled_indices = stochastic_acquisition(bald_scores, acq_size_; acquisition_type="Power")
	elseif al_sampling == "BALD"
        epistemic_uncertainties = pool_prediction_matrix[5, :]
        sampled_indices = top_k_acquisition_no_duplicates(epistemic_uncertainties, acq_size_; descending=true)
    elseif al_sampling == "StochasticBALD"
        bald_scores = pool_prediction_matrix[5, :]
        sampled_indices = stochastic_acquisition(bald_scores, acq_size_; acquisition_type="Stochastic")
    elseif al_sampling == "BayesianEntropicUncertainty$(pct_epistemic_samples)"
        aleatoric_uncertainties = pool_prediction_matrix[4, :]
        epistemic_uncertainties = pool_prediction_matrix[5, :]
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/aleatoric_uncertainties_$(al_step).csv", summary_stats(aleatoric_uncertainties), ',')
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/epistemic_uncertainties_$(al_step).csv", summary_stats(epistemic_uncertainties), ',')
        most_unambiguous_samples = top_k_acquisition_no_duplicates(aleatoric_uncertainties, round(Int, acq_size_ * (1 - pct_epistemic_samples)))
        most_ambiguous_samples = top_k_acquisition_no_duplicates(aleatoric_uncertainties, round(Int, acq_size_ * (1 - pct_epistemic_samples)); descending=true)
        most_uncertain_samples = top_k_acquisition_no_duplicates(epistemic_uncertainties, round(Int, acq_size_ * pct_epistemic_samples); descending=true)
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/most_uncertain_samples_$(al_step).csv", most_uncertain_samples, ',')
		ambiguous_samples_not_in_unceratain_samples = setdiff(most_ambiguous_samples, most_uncertain_samples)
		sampled_indices = vcat(most_uncertain_samples, ambiguous_samples_not_in_unceratain_samples[1:round(Int, acq_size_ * (1 - pct_epistemic_samples))])
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/samplsampled_indices_$(al_step).csv",sampled_indices, ',')
        #saving the uncertainties associated with the queried samples and their labels
        # writedlm("./Experiments/$(experiment)/$(pipeline_name)/predictions/sampled_indices_$(al_step)_stats.csv", [aleatoric_uncertainties[sampled_indices] epistemic_uncertainties[sampled_indices] pool_y[sampled_indices]], ',')
    elseif al_sampling == "PowerEntropy"
        entropy_scores = pool_prediction_matrix[6, :]
        sampled_indices = stochastic_acquisition(entropy_scores, acq_size_; acquisition_type="Power")
    elseif al_sampling == "StochasticEntropy"
        entropy_scores = pool_prediction_matrix[6, :]
        sampled_indices = stochastic_acquisition(entropy_scores, acq_size_; acquisition_type="Stochastic")
    elseif al_sampling == "Diversity"
        error("Diversity Sampling NOT IMPLEMENTED YET")
    elseif al_sampling == "Confidence"
        scores = pool_prediction_matrix[2, :]
        sampled_indices = top_k_acquisition_no_duplicates(scores, acq_size_; descending=false)
    elseif al_sampling == "StdConfidence"
        scores = pool_prediction_matrix[3, :]
        sampled_indices = top_k_acquisition_no_duplicates(scores, acq_size_; descending=true)
	elseif al_sampling == "BayesianUncertainty$(pct_epistemic_samples)"
        aleatoric_uncertainties = pool_prediction_matrix[2, :]
        epistemic_uncertainties = pool_prediction_matrix[5, :]
        most_ambiguous_samples = top_k_acquisition_no_duplicates(aleatoric_uncertainties, acq_size_; descending=false)
        most_uncertain_samples = top_k_acquisition_no_duplicates(epistemic_uncertainties, round(Int, acq_size_ * pct_epistemic_samples); descending=true)
        ambiguous_samples_not_in_unceratain_samples = setdiff(most_ambiguous_samples, most_uncertain_samples)
		sampled_indices = vcat(most_uncertain_samples, ambiguous_samples_not_in_unceratain_samples[1:round(Int, acq_size_ * (1 - pct_epistemic_samples))])
    end
    return sampled_indices
end