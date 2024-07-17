function normalized_entropy10(prob_vec::Vector, n_output::Int64)::Float32
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum(prob_vec) < 0.99
        return error("sum(prob_vec) is not 1 BUT $(sum(prob_vec)) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log10.(prob_vec))) / log10(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::Matrix, n_output::Int64)::Float32
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
function bald(prob_matrix::Matrix, n_output::Int64)::Tuple{Float32, Float32, Float32}
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H - E_H
end

# function variances_ratio(predictions::Vector)
#     count_map = countmap(predictions)
#     # println(count_map)
#     uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
#     index_max = argmax(nUniques)
#     prediction = uniques[index_max]
#     pred_probability = maximum(nUniques) / sum(nUniques)
#     return 1 - pred_probability
# end

