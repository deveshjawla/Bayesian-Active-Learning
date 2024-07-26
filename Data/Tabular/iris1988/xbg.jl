
###
### Data
###

iris = CSV.read("Iris_cleaned.csv", DataFrame, header=1)

target = "Species"
iris = iris[shuffle(axes(iris, 1)), :]
pool, test_set = split_data(iris, at=0.8)
n_input = 4

pool, test_set = pool_test_to_matrix(pool, test_set, n_input)

total_pool_samples = size(pool[1], 2)
input_size = size(pool[1], 1)
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

