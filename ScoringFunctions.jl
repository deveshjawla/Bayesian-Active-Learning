function normalized_entropy(prob_vec::Vector, n_output)
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum(prob_vec) < 0.99
        return error("sum(prob_vec) is not 1 BUT $(sum(prob_vec)) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log2.(prob_vec))) / log2(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)


Returns : H(normalized_entropy), E+H(BALD score, Mutual Information)
"""
function bald(prob_matrix::Matrix, n_output)
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, H + E_H
end

function var_ratio(predictions::Vector)
    count_map = countmap(predictions)
    # println(count_map)
    uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
    index_max = argmax(nUniques)
    prediction = uniques[index_max]
    pred_probability = maximum(nUniques) / sum(nUniques)
    return 1 - pred_probability
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)


Returns : H(normalized_entropy)
"""
function predictive_uncertainty(prob_matrix::Matrix, n_output)
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    return H
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)


Returns : predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty
"""
function uncertainties(prob_matrix::Matrix, n_output)
    # println(size(prob_matrix))
    mean_prob_per_class = vec(mean(prob_matrix, dims=2))
    H = normalized_entropy(mean_prob_per_class, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    # writedlm("./uncertainties.csv", [H E_H])
    return H, E_H, H - E_H
end

