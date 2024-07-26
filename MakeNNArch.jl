function make_nn_arch(nn_arch, input_size, output_size)
    if nn_arch == "Conv"
        nn = Chain(
            # First convolution, operating upon a 28x28 image
            Conv((3, 3), 1 => 8, pad=(1, 1), relu),
            x -> maxpool(x, (2, 2)),
            # Second convolution, operating upon a 14x14 image
            Conv((3, 3), 8 => 8, pad=(1, 1), relu),
            x -> maxpool(x, (2, 2)),
            # Third convolution, operating upon a 7x7 image
            Conv((3, 3), 8 => 8, pad=(1, 1), relu),
            x -> maxpool(x, (2, 2)),
            # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
            # which is where we get the 288 in the `Dense` layer below:
            x -> reshape(x, :, size(x, 4)),
            Dense(72 => 10, relu),
            Dense(10 => output_size),
            # DIR(10 => output_size)
        )
    elseif nn_arch == "Evidential"
        # Define model
        nn = Chain(
            Dense(input_size => 8, relu; bias=true),
            Dense(8 => 8, relu; bias=true),
            DIR(8 => output_size; bias=false)
        )
    elseif nn_arch == "MixActivations4Layers"
        nn = Chain(
            Parallel(vcat, Dense(input_size => input_size, identity; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(input_size => input_size, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * input_size => input_size, identity; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal()), Dense(4 * input_size => input_size, tanh; init=Flux.glorot_normal())),
            Dense(4 * input_size => output_size)
        )
    elseif nn_arch == "DroputNN2Layers"
        nn = Chain(
            Dense(input_size => 5, tanh; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
            Dense(5 => 5, tanh; init=Flux.glorot_normal()),
            # Dropout(dropout_rate),
            Dense(5 => output_size; init=Flux.glorot_normal()),
        )
    end
	return nn
end