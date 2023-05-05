#### 
#### Network - A customizable chain of parametrized conv layer and dense layers. Does not require manually
#### calculating the output size of images. Takes as input batches of images with dims WHCN.
#### Automatically makes the weight matrices which are to be modified by Variational Inference.
#### 

using Flux, Parameters:@with_kw

@with_kw struct DenseParams
    indim::Int = 128
    outdim::Int = 128
    activation_fn = relu
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
    activation_fn = relu
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

conv_1, out_1 = conv_layer(128, (3, 3), 1, 1, relu, (2, 2), bnmom=nothing)
conv_2, out_2 = conv_layer(out_1, (3, 3), 1, 1, relu, (2, 2), bnmom=nothing)
conv_3, out_3 = conv_layer(out_2, (3, 3), 1, 1, relu, (2, 2), bnmom=nothing)
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

dense_layers = [DenseParams(indim = 150, outdim = 5, activation_fn = tanh), DenseParams(indim = 5, outdim = 3, activation_fn = tanh), DenseParams(indim = 3, outdim = 1, activation_fn = sigmoid)]

function forward(x, nn_params, layers_spec)
    nn = Chain([layer(i..., j) for (i, j) in zip(nn_params, layers_spec)]...)
    return nn(x)
end

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
#         Dense(W0, b0, leakyrelu),
#         Dense(W1, b1, leakyrelu),
#         Dense(W2, b2, leakyrelu),
#         Dense(W3, b3, leakyrelu),
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

function nn(theta::AbstractVector)
    W0 = reshape(theta[1:25], 5, 5, 1, 1) # Conv((5, 5), 1=>1, relu)
    b0 = theta[26:26] #same length as the number of output channels
    W1 = reshape(theta[27:35], 3, 3, 1, 1) # Conv((3, 3), 1=>1, relu)
    b1 = theta[36:36]
    W2 = reshape(theta[37:45], 3, 3, 1, 1) # Conv((3, 3), 1=>1, relu)
    b2 = theta[46:46]

    W3 = reshape(theta[1:25], 9, 128)
    b3 = theta[46:46]
    W4 = reshape(theta[1:25], 3, 9)
    b4 = theta[46:46]
    W5 = reshape(theta[1:25], 1, 3)
    b5 = theta[46:46]
	
    model = Chain(
        Conv(W0, b0, relu),
        Conv(W1, b1, relu),
        Conv(W2, b2, relu),
        flatten, # for a defined input image size, we can calculate the flattened size
        Dense(W3, b3, tanh),
        Dense(W4, b4, tanh),
        Dense(W5, b5, sigmoid) # for binary classification
    )
    return model
end

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
#         Dense(W0, b0, leakyrelu),
#         Dense(W1, b1, leakyrelu),
#         Dense(W2, b2, leakyrelu),
#         Dense(W3, b3, leakyrelu),
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