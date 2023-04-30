
###
### Data
###

shap_importances = CSV.read("./shap_importances.csv", DataFrame, header=1)
n_input = 30

pool = CSV.read("train.csv", DataFrame, header=1)
pool[pool.target.==-1, :target] .= 2
pool = select(pool, vcat(shap_importances.feature_name[1:n_input], "target"))
pool = data_balancing(pool, balancing="undersampling", positive_class_label=1, negative_class_label=2)

test = CSV.read("test.csv", DataFrame, header=1)
test[test.target.==-1, :target] .= 2
test = select(test, vcat(shap_importances.feature_name[1:n_input], "target"))
test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)

pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]


