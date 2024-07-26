"""
    DIR(in => out, σ=softplus; bias=true, init=Flux.glorot_uniform)
    DIR(W::AbstractMatrix, [bias], σ)

A Linear layer with a σ=softplus activation function in the end to implement the
Dirichlet evidential distribution. In this layer the number of output nodes
should correspond to the number of classes you wish to model. This layer should
be used to model a Multinomial likelihood with a Dirichlet prior. Thus the
posterior is also a Dirichlet distribution. Moreover the type II maximum
likelihood, i.e., the marginal likelihood is a Dirichlet-Multinomial
distribution. Create a fully connected layer which implements the Dirichlet
Evidential distribution whose forward pass is simply given by:

    y = σ.(W * x .+ bias)

The input `x` should be a vector of length `in`, or batch of vectors represented
as an `in × N` matrix, or any array with `size(x,1) == in`.
The out `y` will be a vector  of length `out`, or a batch with
`size(y) == (out, size(x)[2:end]...)`
The output will have applied the function `softplus(y)` to each row/element of `y`.
Keyword `bias=false` will switch off trainable bias for the layer.
The initialisation of the weight matrix is `W = init(out, in)`, calling the function
given to keyword `init`, with default [`glorot_uniform`](@doc Flux.glorot_uniform).
The weight matrix and/or the bias vector (of length `out`) may also be provided explicitly.

# Arguments:
- `(in, out)`: number of input and output neurons
- `init`: The function to use to initialise the weight matrix.
- `bias`: Whether to include a trainable bias vector.
"""
struct DIR{M<:AbstractMatrix,B,F}
    W::M
    b::B
    σ::F
    function DIR(W::M, b=true, σ::F=softplus) where {M<:AbstractMatrix,F}
        b = Flux.create_bias(W, b, size(W, 1))
        return new{M,typeof(b),F}(W, b, σ)
    end
end

function DIR((in, out)::Pair{<:Integer,<:Integer}, σ=softplus; init=Flux.glorot_uniform, bias=true)
    DIR(init(out, in), bias, σ)
end

Flux.@functor DIR

function (a::DIR)(x::AbstractVecOrMat)
    a.σ.(a.W * x .+ a.b) .+ 1
end

(a::DIR)(x::AbstractArray) = reshape(a(reshape(x, size(x, 1), :)), :, size(x)[2:end]...)