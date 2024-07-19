
# function variances_ratio(predictions::Vector)
#     count_map = countmap(predictions)
#     # println(count_map)
#     uniques, nUniques = collect(keys(count_map)), collect(values(count_map))
#     index_max = argmax(nUniques)
#     prediction = uniques[index_max]
#     pred_probability = maximum(nUniques) / sum(nUniques)
#     return 1 - pred_probability
# end

