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

using SpecialFunctions
# The α here is actually the α̃ which has scaled down evidence that is good.
# the α heres is a matrix of size (K, B) or (O, B)
function kl(α)
    ψ = SpecialFunctions.digamma
    lnΓ = SpecialFunctions.loggamma
    K = first(size(α))
    # Actual computation
    ∑α = sum(α, dims = 1)
    ∑lnΓα = sum(lnΓ.(α), dims = 1)
    A = lnΓ.(∑α) .- lnΓ(K) .- ∑lnΓα
    B = sum((α .- 1) .* (ψ.(α) .- ψ.(∑α)), dims = 1)
    kl = A + B
    kl
end

"""
    dirloss(y, α, t)

Regularized version of a type II maximum likelihood for the Multinomial(p)
distribution where the parameter p, which follows a Dirichlet distribution has
been integrated out.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `α`: the parameters of a Dirichlet distribution representing the belief in each class which shape should be (O, B)
- `t`: counter for the current epoch being evaluated
"""
function dirloss(y, α, t)
    S = sum(α, dims = 1)
    p̂ = α ./ S
    # Main loss
    loss = (y - p̂) .^ 2 .+ p̂ .* (1 .- p̂) ./ (S .+ 1)
    loss = sum(loss, dims = 1)
    # Regularizer
    λₜ = min(1.0, t / 10)
    # Keep only misleading evidence, i.e., penalize stuff that fit badly
    α̂ = @. y + (1 - y) * α
    reg = kl(α̂)
    # Total loss = likelihood + regularizer
    #sum(loss .+ λₜ .* reg, dims = 2)
    sum(loss .+ λₜ .* reg)
end

"""
    dirloss(y, α, t)

Regularized version of a type II maximum likelihood for the Multinomial(p)
distribution where the parameter p, which follows a Dirichlet distribution has
been integrated out.

# Arguments:
- `y`: the targets whose shape should be (O, B)
- `α`: the parameters of a Dirichlet distribution representing the belief in each class which shape should be (O, B)
- `t`: counter for the current epoch being evaluated
"""
function dirlossweighted(y, α, t, sample_weights)
    S = sum(α, dims = 1)
    p̂ = α ./ S
    # Main loss
    loss = (y - p̂) .^ 2 .+ p̂ .* (1 .- p̂) ./ (S .+ 1)
	if size(loss) != size(sample_weights)
        error("dirlossweighted(ŷ, y), size loss = $(size(loss)) and size of sample_weights = $(size(sample_weights)) not the same")
    end
    loss = sum(sample_weights .* loss, dims = 1)
    # Regularizer
    λₜ = min(1.0, t / 10)
    # Keep only misleading evidence, i.e., penalize stuff that fit badly
    α̂ = @. y + (1 - y) * α
    reg = kl(α̂)
    # Total loss = likelihood + regularizer
    #sum(loss .+ λₜ .* reg, dims = 2)
    sum(loss .+ λₜ .* reg)
end


"""
EDLC Uncertainty
"""
edlc_uncertainty(α) = first(size(α)) ./ sum(α, dims=1)