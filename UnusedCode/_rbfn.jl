struct RBF{M<:AbstractArray,B<:AbstractArray}
    W::M #Weights
    b::B #biases
	function RBF(W::M, b) where {M <: AbstractMatrix}
        b = Flux.create_bias(W, ones(size(W, 1)), size(W, 1))
        return new{M, typeof(b)}(W, b)
    end
end

function RBF((in, out)::Pair{<:Integer,<:Integer};
    init=Flux.glorot_uniform, bias=true)
    W = init(out, in)
    return RBF(W, bias)
end

Flux.@functor RBF
function (a::RBF)(x::AbstractArray)
    x2 = sum(abs2, x, dims=1)
    W2 = sum(abs2, a.W, dims=2)
    d = -2 * a.W * x .+ W2 .+ x2
    return exp.(-a.b .* d)
end