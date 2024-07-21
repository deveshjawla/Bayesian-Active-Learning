using StatsBase: countmap
function majority_voting(predictions::AbstractVector)::Vector{Float64}
    count_map = countmap(predictions)
    max_count = 0
    total_count = 0
    prediction = predictions[1]

    for (pred, count) in count_map
        total_count += count
        if count > max_count
            max_count = count
            prediction = pred
        end
    end

    pred_probability = max_count / total_count
    return [prediction, 1 - pred_probability] # 1 - pred_probability => give the uncertainty
end


"""
EDLC Uncertainty
"""
edlc_uncertainty(α) = first(size(α)) ./ sum(α, dims=1)

function normalized_entropy(prob_vec::Vector, n_output)::Float64
    sum_probs = sum(prob_vec)
    if any(i -> i == 0, prob_vec)
        return 0
    elseif n_output == 1
        return error("n_output is $(n_output)")
    elseif sum_probs < 0.99999 || sum_probs > 1.00001
        return error("sum(prob_vec) is not 1 BUT $(sum_probs) and the prob_vector is $(prob_vec)")
    else
        return (-sum(prob_vec .* log.(prob_vec))) / log(n_output)
    end
end

"""
Take Probability Matrix as an argument, whose dimensions are the length of the probability vector, total number of runs(samples from MC Chain, or MC dropout models)

Returns : H(macro_entropy)
"""
function macro_entropy(prob_matrix::Matrix, n_output)::Float32
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
function bald(prob_matrix::Matrix, n_output)
    H = macro_entropy(prob_matrix, n_output)
    E_H = mean(mapslices(x -> normalized_entropy(x, n_output), prob_matrix, dims=1))
    return H, E_H, H - E_H
end