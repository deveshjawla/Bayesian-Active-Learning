
###
### Data
###

shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
n_input = 30

pool = CSV.read("train.csv", DataFrame, header=1)
pool[pool.target.==-1, :target] .= 2
# pool = select(pool, vcat(shap_importances.feature_name[1:n_input], "target"))
pool = balance_binary_data(pool, balancing="undersampling", positive_class_label=1, negative_class_label=2)

test = CSV.read("test.csv", DataFrame, header=1)
test[test.target.==-1, :target] .= 2
# test = select(test, vcat(shap_importances.feature_name[1:n_input], "target"))
# test = balance_binary_data(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

pool, test_set = pool_test_to_matrix(pool, test, n_input)
total_pool_samples = size(pool[1], 2)

n_input = size(pool[1], 1)
n_output = lastindex(unique(pool[2]))

# X=copy(transpose(pool[1]))
# test_x=copy(transpose(test_set[1]))
# Y=vec(Int.(copy(transpose(pool[2])))).-1
# test_y=vec(Int.(copy(transpose(test_set[2])))).-1

pool_x = pool[1]
test_set_x = test_set[1]
pool_y = Int.(pool[2]) .- 1
test_set_y = Int.(test_set[2]) .- 1

pool = (pool_x, pool_y)
test_set = (test_set_x, test_set_y)

