### 
### Data
### 

using DataFrames, DelimitedFiles, Statistics

features = readdlm("Data/SECOM/nan_filtered_data.csv", ',', Float64)
# features = replace(features, NaN => 0)
labels = Int.(readdlm("Data/SECOM/nan_filtered_labels.csv")[:, 1])

# using MultivariateStats

# M = fit(PCA, train_x', maxoutdim = 150)
# train_x_transformed = MultivariateStats.transform(M, train_x')

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, test_x')

# train_x = Matrix(train_x_transformed')
# test_x = Matrix(test_x_transformed')

using MLJ
import XGBoost
import MLJXGBoostInterface

XGB = @load XGBoostClassifier verbosity=0 

# train_y=coerce(DataFrame(real=train_y),:real=>Finite)
# test_y=coerce(DataFrame(real=test_y),:real=>OrderedFactor)
train_y, _ =unpack(coerce(DataFrame(real=train_y),
            :real=>OrderedFactor),
        ==(:real),
        colname -> true, );

test_y, _ =unpack(coerce(DataFrame(real=test_y),
        :real=>OrderedFactor),
    ==(:real),
    colname -> true, );

machbest = machine(
    #     self_tune,
    #     XGB(num_round=30, booster="dart", eta = 1.0, max_depth = 7, rate_drop=0.1),
        XGB(nthread=6, num_round=50, booster="gbtree", eta = 1.0, max_depth = 8,
            scale_pos_weight=13,
            min_child_weight=1,
            ),
            DataFrame(train_x,:auto),train_y
    )

fit!(machbest,verbosity=1 )


predictions = vec(pdf(MLJ.predict(machbest, DataFrame(test_x,:auto) ),[true]))

print("Accuracy:", accuracy([i>0.05 for i in predictions], [i==true for i in test_y]))
print("MCC:", mcc([i>0.005 for i in predictions], [i==true for i in test_y]))

ConfusionMatrix()([i>0.05 for i in predictions], [i==true for i in test_y])
