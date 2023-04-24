function entropy(prob_vec::Vector, n_output)
	return (-sum(prob_vec.*log2.(prob_vec)))/log2(n_output)
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)


Returns : H(Entropy), E+H(BALD score, Mutual Information)
"""
function bald(prob_matrix::Matrix, n_output)
	# println(size(prob_matrix))
	mean_prob_per_class = vec(mean(prob_matrix, dims = 2))
	H = entropy(mean_prob_per_class, n_output)
	E_H = mean(mapslices(x->entropy(x, n_output), prob_matrix, dims=1))
	return H, H + E_H
end

function var_ratio(predictions::Vector)
    count_map = countmap(predictions)
    # println(count_map)
    uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
    index_max = argmax(nUniques)
    prediction = uniques[index_max]
    pred_probability = maximum(nUniques) / sum(nUniques)
	return 1-pred_probability
end