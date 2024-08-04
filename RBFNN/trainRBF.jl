"""
Builds an RBF Network from the provided training set.
[Centers, betas, Theta] = trainRBFN(X_train, y_train, centersPerCategory, verbose)

There are three main steps to the training process:
1. Prototype selection through k-means clustering.
2. Calculation of beta coefficient (which controls the width of the 
RBF neuron activation function) for each RBF neuron.
3. Training of output weights for each category using gradient descent.

Parameters
X_train  - The training vectors, one per row
y_train  - The category values for the corresponding training vector.
Category values should be continuous starting from 1. (e.g.,
1, 2, 3, ...)
centersPerCategory - How many RBF centers to select per category. k-Means
requires that you specify 'k', the number of 
clusters to look for.
verbose  - Whether to print out messages about the training status.

Returns
Centers  - The prototype vectors stored in the RBF neurons.
betas    - The beta coefficient for each coressponding RBF neuron.
Theta    - The weights for the output layer. There is one row per neuron
and one column per output node / category.
"""
function trainRbf(X_train::Array{Float32,2}, y_train::Vector{Int64}, centersPerCategory::Int64, verbose::Bool, numCats::Int)
    # Get the number of unique categories in the dataset.
    # numCats = size(unique(y_train), 1)

    # Ensure category values are non-zero and continuous.
    # This allows the index of the output node to equal its category (e.g.,
    # the first output node is category 1).
    if (any(y_train .== 0) || any(y_train .> numCats))
        error("Category values must be non-zero and continuous.")
    end

    # ================================================
    #       Select RBF Centers and Parameters
    # ================================================

    Centers, betas, numRBFNeurons = computeCentroids(X_train, y_train, centersPerCategory, numCats, verbose)

    # ==========================================================
    #       Compute RBF Activations Over The Training Set
    # ===========================================================

    X_activ = computeRBFActivations(X_train, Centers, betas, numRBFNeurons, verbose)

    # =============================================
    #       Train Output Weights by performing Gradient Descent
    # =============================================

    Theta, nn = computeWeights(X_activ, y_train, numRBFNeurons, numCats, verbose)

    Centers, betas, Theta, nn
end