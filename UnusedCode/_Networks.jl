#### 
#### Network - A customizable chain of parametrized conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### Automatically makes the weight matrices which are to be modified by Variational Inference.
#### 

using Flux, Parameters:@with_kw

@with_kw struct DenseParams
    indim::Int = 128
    outdim::Int = 128
    activation_fn = tanh
    bnmom::Union{Float32,Nothing} = nothing
    bias::Bool = false
end

"""
# Function to make a custom dense layer
"""
function layer(weight::AbstractArray, bias::AbstractArray, dp::DenseParams)
    if dp.bnmom === nothing
        Dense(weight, bias, dp.activation_fn)
    else
        Chain(
            Dense(weight, bias),
            BatchNorm(dp.outdim, dp.activation_fn, momentum=dp.bnmom)
        )
    end
end

@with_kw struct ConvParams
    filter_size::Tuple{Int,Int} = (3, 3)
    in_channels::Int = 1
    out_channels::Int = 1
    activation_fn = tanh
    stride_length::Int = 1
    pad::Int = 1
    pool_window::Tuple{Int,Int} = (2, 2)
    pool_stride::Int = 1
    bnmom::Union{Float32,Nothing} = nothing
end

"""
# Fucntion to make a custom Covolution layer.
## Returns a tuple, (layer, output_image_dims)
"""
function layer(weight::AbstractArray, bias::AbstractArray, cp::ConvParams)
    if cp.bnmom === nothing
        layer = Chain(
            Conv(weight, bias, cp.activation_fn, pad=cp.pad),
            x -> meanpool(x, cp.pool_window, stride=cp.pool_stride)
        )
    else
        layer = Chain(
            Conv(weight, bias, pad=cp.pad),
            x -> meanpool(x, cp.pool_window, stride=cp.pool_stride),
            BatchNorm(cp.in_channels, cp.activation_fn, momentum=cp.bnmom)
        )
    end
    return layer
end


#### 
#### Network - A customizable chain of conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### 

using Flux, Statistics
using Printf, BSON

"""
# Function to make a custom dense layer
"""
function dense_layer(indim::Int, outdim::Int,
    bnmom::Float32, activation_fn)

    if typeof(bnmom) == Float32
        Chain(
            Dense(indim, outdim),
            BatchNorm(outdim, activation_fn, momentum=bnmom)
        )
    else
        Dense(indim, outdim, activation_fn)
    end
end

"""
# Fucntion to make a custom Covolution layer.
## Returns a tuple, (layer, output_image_dims)
"""
function conv_layer(input_dims, filter_size::Tuple{Int,Int},
    i::Int, o::Int, activation_fn, pool_window::Tuple{Int,Int};
    stride_length=1, pool_stride=1, bnmom=0.5)

    pad = filter_size .÷ 2
    if typeof(bnmom) == Float32
        layer = Chain(
            Conv(filter_size, i => o, activation_fn, pad=pad),
            x -> meanpool(x, pool_window),
            BatchNorm(i, activation_fn, momentum=bnmom)
        )
    else
        layer = Chain(
            Conv(filter_size, i => o, activation_fn, pad=pad),
            x -> meanpool(x, pool_window, stride=pool_stride)
        )
    end
    output_dims_conv = (((input_dims[1] - filter_size[1] + 2 * pad[1]) / stride_length) + 1)
    output_dims = ((output_dims_conv - pool_window[1]) / pool_stride) + 1
    return layer, Int(output_dims)
end

conv_1, out_1 = conv_layer(128, (3, 3), 1, 1, tanh, (2, 2), bnmom=nothing)
conv_2, out_2 = conv_layer(out_1, (3, 3), 1, 1, tanh, (2, 2), bnmom=nothing)
conv_3, out_3 = conv_layer(out_2, (3, 3), 1, 1, tanh, (2, 2), bnmom=nothing)
dense_1 = dense_layer(out_3^2, 128, 0.1f0, relu)
dense_2 = dense_layer(128, 10, 0.1f0, relu)

# An example Network. You may customize each layer above or remove/add layers as needed.
network = Chain(conv_1, conv_2, conv_3, flatten, dense_1, dense_2, softmax)
# network(rand(128,128,1,1))


###
### Helper Functions
###

"""
Returns the output dimesions(except the last, batch size) of a convolution layer
"""
function conv_out_dims(input_dims::Tuple, cp::ConvParams)
    output_dims_conv = (((input_dims[1] - cp.filter_size[1] + 2 * cp.pad) / cp.stride_lastindex) + 1)
    output_dims = ((output_dims_conv - cp.pool_window[1]) / cp.pool_stride) + 1
    return (Int(output_dims), Int(output_dims))
end

function num_params(cp::ConvParams)
    return prod([cp.filter_size..., cp.in_channels, cp.out_channels]) + cp.out_channels
end

function num_params(dp::DenseParams)
    return (dp.indim * dp.outdim) + dp.outdim
end

"""
Returns a tuple of the weight and bias arrays
"""
function layer_params(params_vec::AbstractVector, cp::ConvParams)
    bias = [pop!(params_vec) for _ in 1:cp.out_channels]
    weight = reshape(params_vec, cp.filter_size..., cp.in_channels, cp.out_channels)
    return weight, bias
end

function layer_params(params_vec::AbstractVector, dp::DenseParams)
    if dp.bias == true
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
    else
        bias = [pop!(params_vec) for _ in 1:dp.outdim]
        # bias .*= 0.0
    end
    weight = reshape(params_vec, dp.outdim, dp.indim)
    return weight, bias
end

"""
Splits a vector x at indices n
"""
function split(x::AbstractVector, n)
    result = Vector{Vector{eltype(x)}}(undef, lastindex(n))
    sum_elements = sum(n)
    if sum_elements == lastindex(x)
        for i in 1:lastindex(n)
            result[i] = splice!(x, 1:n[i])
        end
    end
    return result
end

"""
Make a list of tuples of layer parameters i.e [(W1,b1),...]
"""
function unpack_params(nn_params::AbstractVector, layers_spec::AbstractVector)
    num_params_list = num_params.(layers_spec)
    indices_list = split(nn_params, num_params_list)
    params_collection = [layer_params(i, j) for (i, j) in zip(indices_list, layers_spec)]
    return params_collection
end



###
### Conv Network specifications
###

# conv_layers = [ConvParams(), ConvParams(), ConvParams()]

# input_size = (128, 128)
# final_conv_out_dims = conv_out_dims(input_size, conv_layers[1]) |> x -> conv_out_dims(x, conv_layers[2]) |> x -> conv_out_dims(x, conv_layers[3])

# dense_layers = [DenseParams(indim = prod([final_conv_out_dims..., conv_layers[end].out_channels])), DenseParams(outdim = 10)]

# layers_spec = [conv_layers; dense_layers]

# function forward(x, nn_params::AbstractVector, layers_spec)
#     c1, c2, c3, d1, d2 = unpack_params(nn_params, layers_spec)
#     nn = Chain([layer(i..., j) for (i, j) in zip([c1, c2, c3], layers_spec[1:3])]..., Flux.flatten, [layer(i..., j) for (i, j) in zip([d1, d2], layers_spec[4:5])]..., softmax)
#     return nn(x)
# end

###
### Dense Network specifications
###

# dense_layers = [DenseParams(indim = 150, outdim = 5, activation_fn = relu), DenseParams(indim = 5, outdim = 3, activation_fn = relu), DenseParams(indim = 3, outdim = 1, activation_fn = sigmoid)]

# function forward(x, nn_params, layers_spec)
#     nn = Chain([layer(i..., j) for (i, j) in zip(nn_params, layers_spec)]...)
#     return nn(x)
# end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:45400], 100, 454)
#     b0 = θ[45401:45500]
#     W1 = reshape(θ[45501:55500], 100, 100)
#     b1 = θ[55501:55600]
#     W2 = reshape(θ[55601:57600], 20, 100)
#     b2 = θ[57601:57620]
#     W3 = reshape(θ[57621:58020], 20, 20)
#     b3 = θ[58021:58040]
#     W4 = reshape(θ[58041:58060], 1, 20)
#     b4 = θ[58061:58061]
#     model = Chain(
#         Dense(W0, b0, leakymish),
#         Dense(W1, b1, leakymish),
#         Dense(W2, b2, leakymish),
#         Dense(W3, b3, leakymish),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:3000], 100, 30)
#     b0 = θ[3001:3100]
#     W1 = reshape(θ[3101:13100], 100, 100)
#     b1 = θ[13101:13200]
#     W2 = reshape(θ[13201:15200], 20, 100)
#     b2 = θ[15201:15220]
#     W3 = reshape(θ[15221:15620], 20, 20)
#     b3 = θ[15621:15640]
#     W4 = reshape(θ[15641:15660], 1, 20)
#     b4 = θ[15661:15661]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, σ)
#     )
#     return model
# end


# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:600], 100, 6)
#     b0 = θ[601:700]
#     W1 = reshape(θ[701:10700], 100, 100)
#     b1 = θ[10701:10800]
#     W2 = reshape(θ[10801:12800], 20, 100)
#     b2 = θ[12801:12820]
#     W3 = reshape(θ[12821:13220], 20, 20)
#     b3 = θ[13221:13240]
#     W4 = reshape(θ[13241:13260], 1, 20)
#     b4 = θ[13261:13261]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:1000], 100, 10)
#     b0 = θ[1001:1100]
#     W1 = reshape(θ[1101:11100], 100, 100)
#     b1 = θ[11101:11200]
#     W2 = reshape(θ[11201:13200], 20, 100)
#     b2 = θ[13201:13220]
#     W3 = reshape(θ[13221:13620], 20, 20)
#     b3 = θ[13621:13640]
#     W4 = reshape(θ[13641:13660], 1, 20)
#     b4 = θ[13661:13661]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end



###
### Conv Network specifications
###

using Flux

# function nn(theta::AbstractVector)
#     W0 = reshape(theta[1:25], 5, 5, 1, 1) # Conv((5, 5), 1=>1, relu)
#     b0 = theta[26:26] #same length as the number of output channels
#     W1 = reshape(theta[27:35], 3, 3, 1, 1) # Conv((3, 3), 1=>1, relu)
#     b1 = theta[36:36]
#     W2 = reshape(theta[37:45], 3, 3, 1, 1) # Conv((3, 3), 1=>1, relu)
#     b2 = theta[46:46]

#     W3 = reshape(theta[1:25], 9, 128)
#     b3 = theta[46:46]
#     W4 = reshape(theta[1:25], 3, 9)
#     b4 = theta[46:46]
#     W5 = reshape(theta[1:25], 1, 3)
#     b5 = theta[46:46]
	
#     model = Chain(
#         Conv(W0, b0, relu),
#         Conv(W1, b1, relu),
#         Conv(W2, b2, relu),
#         flatten, # for a defined input image size, we can calculate the flattened size
#         Dense(W3, b3, relu),
#         Dense(W4, b4, relu),
#         Dense(W5, b5, sigmoid) # for binary classification
#     )
#     return model
# end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:45400], 100, 454)
#     b0 = θ[45401:45500]
#     W1 = reshape(θ[45501:55500], 100, 100)
#     b1 = θ[55501:55600]
#     W2 = reshape(θ[55601:57600], 20, 100)
#     b2 = θ[57601:57620]
#     W3 = reshape(θ[57621:58020], 20, 20)
#     b3 = θ[58021:58040]
#     W4 = reshape(θ[58041:58060], 1, 20)
#     b4 = θ[58061:58061]
#     model = Chain(
#         Dense(W0, b0, leakymish),
#         Dense(W1, b1, leakymish),
#         Dense(W2, b2, leakymish),
#         Dense(W3, b3, leakymish),
#         Dense(W4, b4, sigmoid)
#     )
#     return model
# end

# function feedforward(θ::AbstractVector)
#     W0 = reshape(θ[1:3000], 100, 30)
#     b0 = θ[3001:3100]
#     W1 = reshape(θ[3101:13100], 100, 100)
#     b1 = θ[13101:13200]
#     W2 = reshape(θ[13201:15200], 20, 100)
#     b2 = θ[15201:15220]
#     W3 = reshape(θ[15221:15620], 20, 20)
#     b3 = θ[15621:15640]
#     W4 = reshape(θ[15641:15660], 1, 20)
#     b4 = θ[15661:15661]
#     model = Chain(
#         Dense(W0, b0, relu),
#         Dense(W1, b1, relu),
#         Dense(W2, b2, relu),
#         Dense(W3, b3, relu),
#         Dense(W4, b4, σ)
#     )
#     return model
# end


    # function feedforward(θ::AbstractVector)
    #     W0 = reshape(θ[1:60], 10, 6)
    #     b0 = θ[61:70]
    #     W1 = reshape(θ[71:170], 10, 10)
    #     b1 = θ[171:180]
    #     W2 = reshape(θ[181:280], 10, 10)
    #     b2 = θ[281:290]
    #     W3 = reshape(θ[291:390], 10, 10)
    #     b3 = θ[391:400]
    #     W4 = reshape(θ[401:420], 2, 10)
    #     b4 = θ[421:422]
    #     model = Chain(
    #         Dense(W0, b0, relu),
    #         Dense(W1, b1, relu),
    #         Dense(W2, b2, relu),
    #         Dense(W3, b3, relu),
    #         Dense(W4, b4),
    #         softmax
    #     )
    #     return model
    # end


using Flux, Turing
	# Specify the network architecture.
network_shape = [
    (23,23, :relu),
    (23,23, :relu), 
    (2,23, :σ)]

# Regularization, parameter variance, and total number of
# parameters.
alpha = 0.09
sig = sqrt(1.0 / alpha)
num_params = sum([i * o + i for (i, o, _) in network_shape])

# This modification of the unpack function generates a series of vectors
# given a network shape.
# function unpack(θ::AbstractVector, network_shape::AbstractVector)
#     index = 1
#     weights = []
#     biases = []
#     for layer in network_shape
#         rows, cols, _ = layer
#         size = rows * cols
#         last_index_w = size + index - 1
#         last_index_b = last_index_w + rows
#         push!(weights, reshape(θ[index:last_index_w], rows, cols))
#         push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
#         index = last_index_b + 1
#     end
#     return weights, biases
# end

function feedforward(θ::AbstractVector, network_shape::AbstractVector)
    index = 1
    weights = []
    biases = []
    for layer in network_shape
        rows, cols, _ = layer
        size = rows * cols
        last_index_w = size + index - 1
        last_index_b = last_index_w + rows
        push!(weights, reshape(θ[index:last_index_w], rows, cols))
        push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
        index = last_index_b + 1
    end
    layers = []
    for i in eachindex(network_shape)
        push!(layers, Dense(weights[i],
            biases[i],
            eval(network_shape[i][3])))
    end
    return Chain(layers..., softmax)
end

# Generate an abstract neural network given a shape, 
# and return a prediction.
# function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
#     weights, biases = unpack(θ, network_shape)
#     layers = []
#     for i in eachindex(network_shape)
#         push!(layers, Dense(weights[i],
#             biases[i],
#             eval(network_shape[i][3])))
#     end
#     nn = Chain(layers...)
#     return nn(x)
# end


# General Turing specification for a BNN model.
@model bayes_nn_general(xs, ts, network_shape, num_params) = begin
    θ ~ MvNormal(zeros(num_params), sig .* ones(num_params))
    preds = nn_forward(xs, θ, network_shape)
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end
@model bayes_nn_general(xs, ts, network_shape, num_params) = begin
    θ ~ MvNormal(zeros(num_params), sig .* ones(num_params))
    preds = nn_forward(θ, network_shape)(xs)
    for i = 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

# Set the backend.
Turing.setadbackend(:reverse_diff)

# Perform inference.
num_samples = 500
ch2 = sample(bayes_nn_general(hcat(xs...), ts, network_shape, num_params), NUTS(0.65), num_samples);

# input_size = $input_size
#     n_output = $n_output
l1, l2, l3, l4 = 5, 5, 5, 5
nl1 = input_size * l1 + l1
nl2 = l1 * l2 + l2
nl3 = l2 * l3 + l3
nl4 = l3 * l4 + l4
n_output_layer = l4 * n_output + n_output

total_num_params = nl1 + nl2 + nl3 + nl4 + n_output_layer

# function feedforward(θ::AbstractVector)
# 	W0 = reshape(θ[1:20], 5, 4)
# 	b0 = θ[21:25]
# 	W1 = reshape(θ[26:50], 5, 5)
# 	b1 = θ[51:55]
# 	W2 = reshape(θ[56:80], 5, 5)
# 	b2 = θ[81:85]
# 	W3 = reshape(θ[86:110], 5, 5)
# 	b3 = θ[111:115]
# 	W4 = reshape(θ[116:130], 3, 5)
# 	b4 = θ[131:133]
# 	model = Chain(
# 		Dense(W0, b0, relu),
# 		Dense(W1, b1, relu),
# 		Dense(W2, b2, relu),
# 		Dense(W3, b3, relu),
# 		Dense(W4, b4),
# 		softmax
# 	)
# 	return model
# end

nn_initial = Chain(Dense(input_size, l1, relu), Dense(l1, l2, relu), Dense(l2, l3, relu), Dense(l3, l4, relu), Dense(l4, n_output, relu), softmax)

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, feedforward = Flux.destructure(nn_initial)

total_num_params = length(parameters_initial) # number of paraemters in NN



# function feedforward(θ::AbstractVector)
# 	W0 = reshape(θ[1:600], 20, 30)
# 	b0 = θ[601:620]
# 	W1 = reshape(θ[621:1020], 20, 20)
# 	b1 = θ[1021:1040]
# 	W2 = reshape(θ[1041:1440], 20, 20)
# 	b2 = θ[1441:1460]
# 	W3 = reshape(θ[1461:1860], 20, 20)
# 	b3 = θ[1861:1880]
# 	W4 = reshape(θ[1881:1920], 2, 20)
# 	b4 = θ[1921:1922]
# 	model = Chain(
# 		Dense(W0, b0, relu),
# 		Dense(W1, b1, relu),
# 		Dense(W2, b2, relu),
# 		Dense(W3, b3, relu),
# 		Dense(W4, b4),
# 		softmax
# 	)
# 	return model
# end

# function feedforward(θ::AbstractVector)
# 	W0 = reshape(θ[1:200], 20, 10)
# 	b0 = θ[201:220]
# 	W1 = reshape(θ[221:620], 20, 20)
# 	b1 = θ[621:640]
# 	W2 = reshape(θ[641:1040], 20, 20)
# 	b2 = θ[1041:1060]
# 	W3 = reshape(θ[1061:1460], 20, 20)
# 	b3 = θ[1461:1480]
# 	W4 = reshape(θ[1481:1520], 2, 20)
# 	b4 = θ[1521:1522]
# 	model = Chain(
# 		Dense(W0, b0, relu),
# 		Dense(W1, b1, relu),
# 		Dense(W2, b2, relu),
# 		Dense(W3, b3, relu),
# 		Dense(W4, b4),
# 		softmax
# 	)
# 	return model
# end



