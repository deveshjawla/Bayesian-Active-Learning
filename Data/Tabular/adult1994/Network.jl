# model = Chain(Dense(4 => 12, relu), Dense(12 => 12, relu), Dense(12 => 2, bias=false))

# _, feedforward = Flux.destructure(model)

# l1, l2 = 12, 12
# nl1 = 4 * l1 + l1
# nl2 = l1 * l2 + l2
# n_output_layer = l2 * n_output

# total_num_params = nl1 + nl2 + n_output_layer

# function feedforward(nn_params::AbstractVector)
#     w10 = reshape(nn_params[1:48], 12, 4)
#     b10 = reshape(nn_params[49:60], 12)

#     w20 = reshape(nn_params[61:204], 12, 12)
#     b20 = reshape(nn_params[205:216], 12)

#     w31 = reshape(nn_params[217:240], 2, 12)

#     model = Chain(Dense(w10, b10, relu), Dense(w20, b20, relu), Dense(w31, false))
#     return model
# end



# function feedforward(nn_params::AbstractVector)
#     w10 = reshape(nn_params[1:50], 1, 50)
#     b10 = reshape(nn_params[51:51], 1)

#     w31 = reshape(nn_params[52:53], 2, 1)

#     model = Chain(Dense(w10, b10, relu), Dense(w31, false))
#     return model
# end

# num_params = 53


# # using Flux

# function feedforward(nn_params::AbstractVector, input_size::Int, layer_sizes::Vector{Int}, output_size::Int)
#     """
#     Constructs a feedforward neural network with arbitrary input size and customizable hidden layers.

#     Parameters:
#     	- nn_params: A vector of parameters (weights and biases) flattened.
#     	- input_size: Number of input features.
#     	- layer_sizes: A vector specifying the number of neurons in each hidden layer.
#     	- output_size: The number of output neurons (e.g., for classification or regression).
#     """
#     # Initialize index to keep track of where we are in nn_params
#     idx = 1

#     # Create list of Dense layers
#     layers = []

#     # Handle first layer (input layer to first hidden layer)
#     first_layer_neurons = layer_sizes[1]
#     w1_size = first_layer_neurons * input_size
#     b1_size = first_layer_neurons

#     # Extract weights and biases for the first layer
#     w1 = reshape(nn_params[idx:idx+w1_size-1], first_layer_neurons, input_size)
#     idx += w1_size
#     b1 = reshape(nn_params[idx:idx+b1_size-1], first_layer_neurons)
#     idx += b1_size

#     push!(layers, Dense(w1, b1, relu))

#     # Handle the hidden layers (between input and output)
#     for i in 2:length(layer_sizes)
#         prev_layer_neurons = layer_sizes[i-1]
#         current_layer_neurons = layer_sizes[i]

#         w_size = current_layer_neurons * prev_layer_neurons
#         b_size = current_layer_neurons

#         # Extract weights and biases
#         w = reshape(nn_params[idx:idx+w_size-1], current_layer_neurons, prev_layer_neurons)
#         idx += w_size
#         b = reshape(nn_params[idx:idx+b_size-1], current_layer_neurons)
#         idx += b_size

#         push!(layers, Dense(w, b, relu))
#     end

#     # Handle the output layer
#     last_hidden_neurons = layer_sizes[end]
#     w_out_size = output_size * last_hidden_neurons

#     # Extract weights for the output layer (no bias or different activation function)
#     w_out = reshape(nn_params[idx:idx+w_out_size-1], output_size, last_hidden_neurons)

#     # Create the model (chain of layers)
#     model = Chain(layers..., Dense(w_out, false))  # No bias on the output layer

#     return model
# end

# input_size = 4                   # Input size (e.g., 4 features)
# layer_sizes = [12, 12]            # Two hidden layers with 12 neurons each
# output_size = 2                   # Two output neurons

# # Number of parameters for weights and biases
# n_params = input_size * layer_sizes[1] + layer_sizes[1] +   # First layer
#            layer_sizes[1] * layer_sizes[2] + layer_sizes[2] + # Second layer
#            layer_sizes[2] * output_size                     # Output layer

# # Generate random parameters for the example
# nn_params = randn(n_params)

# # Instantiate the model
# model = feedforward(nn_params, input_size, layer_sizes, output_size)

