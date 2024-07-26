"""
This function computes the activation values of all of the RBFN neurons
for a given batch of input data X_train, provided 'Centers' and their 'betas'

Parameters
Centers  - The prototype vectors for the RBF neurons.
betas    - The beta coefficients for the corresponding prototypes.

Returns
A matrix representing the activations of the RBF Neurons with dimenstions (n_rbf_neurons, n_data_samples)
"""
function computeRBFActivations(X_train::Array{Float32,2}, Centers::Array{Float32,2}, betas::Array{Float32,2}, numRBFNeurons::Int64, verbose::Bool)
    if (verbose)
        print("\n2. Calculate RBF neuron activations over full training set, where Number of RBF neurons = $numRBFNeurons\n")
    end

    zs = mapslices(x -> exp.(-betas .* x), pairwise(SqEuclidean(), Centers, X_train, dims=1), dims=1)

    return zs
end

"""
This function computes the RBF activation function for each of the RBF 
neurons given the supplied 'input'. Each RBF neuron is described by a 
prototype or "center" vector in 'centers', and a beta coefficient in
'betas'. 

Parameters
centers  - Matrix of RBF neuron center vectors, one per row.
betas    - Vector of beta coefficients for the corresponding RBF neuron.
input    - Column vector containing the input.

Returns
A column vector containing the activation value (between 0 and 1) for
each of the RBF neurons.
"""
function getRBFActivations(centers::Array{Float32,2}, betas::Array{Float32,2}, input::Array{Float32,2})
    # Calculate Squared Euclidean Distance between all the centers and the single input data-point
    sqrdDists = pairwise(SqEuclidean(), centers, input, dims=1)
	z = exp.(-betas .* sqrdDists)
    return z
end
