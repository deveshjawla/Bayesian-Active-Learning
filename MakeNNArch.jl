function make_nn_arch(nn_arch::String, n_input::Int, n_output::Int; dropout_rate=0.2)
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
            Dense(10 => n_output; bias=false,)
        )
    elseif nn_arch == "Evidential Classification"
        # Define model
        nn = Chain(
            Dense(n_input => 8, relu; bias=true),
            Dense(8 => 8, relu; bias=true),
            DIR(8 => n_output; bias=false)
        )
    elseif nn_arch == "MixActivations4Layers"
        nn = Chain(
            Parallel(vcat, Dense(n_input => n_input, identity; init=Flux.glorot_normal()), Dense(n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(n_input => n_input, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * n_input => n_input, identity; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * n_input => n_input, identity; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal())),
            Parallel(vcat, Dense(4 * n_input => n_input, identity; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal()), Dense(4 * n_input => n_input, tanh; init=Flux.glorot_normal())),
            Dense(4 * n_input => n_output; bias=false)
        )
    elseif nn_arch == "DroputNN2Layers"
        nn = Chain(
            Dense(n_input => 5, tanh; init=Flux.glorot_normal()),
            Dropout(dropout_rate),
            Dense(5 => 5, tanh; init=Flux.glorot_normal()),
            Dropout(dropout_rate),
            Dense(5 => n_output; bias=false, init=Flux.glorot_normal()),
        )
    elseif nn_arch == "Relu2Layers"
        nn = Chain(
            Dense(n_input => 5, relu; init=Flux.glorot_normal()),
            Dense(5 => 5, relu; init=Flux.glorot_normal()),
            Dense(5 => n_output; bias=false, init=Flux.glorot_normal()),
        )
    end
    return nn
end