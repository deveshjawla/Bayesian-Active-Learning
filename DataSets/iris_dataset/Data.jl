###
### Data
###
iris = CSV.read("Iris_cleaned.csv", DataFrame, header=1)

target = "Species"
using Random
iris = iris[shuffle(axes(iris, 1)), :]
pool, test = split_data(iris, at=0.8)
n_input = 4

pool, test = pool_test_maker(pool, test, n_input)
total_pool_samples = size(pool[1])[2]

