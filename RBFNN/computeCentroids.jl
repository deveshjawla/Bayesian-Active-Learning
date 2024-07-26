"""
Select RBF Centers and Parameters
================================================
Here I am selecting the cluster centers using k-Means clustering.
I've chosen to separate the data by category and cluster each category
separately, though I've read that this step is often done over the full
unlabeled dataset. I haven't compared the accuracy of the two approaches.

Centers = [    -0.4528   -0.7259;
1.1624   -0.2929;
0.8586   -1.3844;
-0.8102    0.3913;
0.4574    0.3437;
-0.2140    1.2991;
-1.4650    0.7938;
-1.3791   -0.8765;
1.0840    1.0717;
0.8385   -1.1455]

betas =
[2.2308;
3.7955;
3.0337;
3.3659;
3.3811;
2.3574;
2.7513;
2.3913;
2.4260;
0.8062]'
betas = betas'
"""
function computeCentroids(X_train::Array{Float32,2}, y_train::Array{Int64,1}, centersPerCategory::Int64, numCats::Int64, verbose::Bool)::Tuple{Array{Float32,2},Array{Float32,2},Int64}
    if (verbose)
        println("1. Selecting centers through k-Means.\n")
    end

    numDims = size(X_train, 2)
    Centers = rand(0, numDims)#[];
    betas = rand(0, 1)#[]    

    # For each of the categories...
    for c = 1:numCats

        if (verbose)
            @printf("  Category %d centers...\n", c)
        end

        # Select the training vectors for category 'c'.
        Xc = X_train[(y_train.==c), :]

        # ================================
        #      Find cluster centers
        # ================================

        # Pick the first 'centersPerCategory' samples to use as the initial centers.
        init_Centroids = Xc[1:centersPerCategory, :]

        # Run k-means clustering, with at most 100 iterations.        
        @printf("  Running kmeans with %d centers...\n", centersPerCategory)
        result = kmeans(permutedims(Xc), centersPerCategory)
        Centroids_c = permutedims(result.centers)
        memberships_c = result.assignments

        # Remove any empty clusters.
        #toRemove = [];
        # 
        # For each of the centroids...
        #for (i = 1 : size(Centroids_c, 1))
        #    # If this centroid has no members, mark it for removal.
        #    if (sum(memberships_c .== i) == 0)        
        #        toRemove = [toRemove; i];
        #    end
        #end
        #
        # If there were empty clusters...
        #if (~isempty(toRemove))
        #    # Remove the centroids of the empty clusters.
        #    Centroids_c(toRemove, :) = [];
        #    
        #    # Reassign the memberships (index values will have changed).
        #    memberships_c = findClosestCentroids(Xc, Centroids_c);
        #end

        # ================================
        #    Compute Beta Coefficients
        # ================================
        if (verbose)
            @printf("  Category %d betas...\n", c)
        end

        # Compute betas for all the clusters.
        betas_c = computeRBFBetas(Xc, Centroids_c, memberships_c)

        # Add the centroids and their beta values to the network.
        Centers = vcat(Centers, Centroids_c)
        betas = vcat(betas, betas_c)
    end
    numRBFNeurons = size(Centers, 1)
    return Centers, betas, numRBFNeurons
end