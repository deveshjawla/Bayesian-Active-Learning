using Distributions

function random_acquisition(pool_scores::Vector, acquisition_size)
    indices_list = collect(1:lastindex(pool_scores))
    shuffled = shuffle(indices_list)
    return shuffled[1:acquisition_size]
end

function initial_random_acquisition(initial_pool_size, acquisition_size)
    indices_list = collect(1:initial_pool_size)
    shuffled = shuffle(indices_list)
    return shuffled[1:acquisition_size]
end

function top_k_acquisition(pool_scores::Vector, acquisition_size; descending=false, remove_zeros=false)
    df = DataFrame(Scores=pool_scores, Sample_indices=collect(1:lastindex(pool_scores)))
    sorted_df = sort(df, :Scores, rev = descending)
	if remove_zeros
		n_non_zero_scores = count(x->x>0.0001, sorted_df.Scores)
		if n_non_zero_scores >= acquisition_size
			top_k = sorted_df[1:acquisition_size, :Sample_indices]
		else
			top_k = sorted_df[1:n_non_zero_scores, :Sample_indices]
		end
	else
		top_k = sorted_df[1:acquisition_size, :Sample_indices]
	end
    return top_k
end

function softmax_acquisition(pool_scores::Vector, acquisition_size; β=1.0)
	gumbel_dist = Gumbel(0, β^-1)
	scores=pool_scores .+ rand(gumbel_dist, lastindex(pool_scores))
	df = DataFrame(Scores=scores, Sample_indices=collect(1:lastindex(scores)))
	sorted_df = sort(df, :Scores)
	return sorted_df[1:acquisition_size, :Sample_indices]
end

function power_acquisition(pool_scores::Vector, acquisition_size; β=1.0)
	gumbel_dist = Gumbel(0, β^-1)
    scores = log.(pool_scores) .+ rand(gumbel_dist, lastindex(pool_scores))
    df = DataFrame(Scores=scores, Sample_indices=collect(1:lastindex(scores)))
	# println(log.(pool_scores))
    sorted_df = sort(df, :Scores, rev=true)
	# println(sorted_df)
    return sorted_df[1:acquisition_size, :Sample_indices]
end