l1, l2 = 4, 4
nl1 = 4 * l1 + l1
nl2 = l1 * l2 + l2
n_output_layer = l2 * n_output + n_output

total_num_params = nl1 + nl2 + n_output_layer

function feedforward(θ::AbstractVector)
	W0 = reshape(θ[1:16], 4, 4)
	b0 = θ[17:20]
	W1 = reshape(θ[21:36], 4, 4)
	b1 = θ[37:40]
	W2 = reshape(θ[41:48], 2, 4)
	b2 = θ[49:50]

	model = Chain(
		Dense(W0, b0, mish),
		Dense(W1, b1, mish),
		Dense(W2, b2),
		softmax
	)
	return model
end

num_params = 50


# From worker 2:    1 ─ %1  = (1:484)::Core.Const(1:484)
# From worker 2:    │   %2  = Base.getindex(θ, %1)::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 2:    │         (W0 = Main.reshape(%2, 22, 22))
# From worker 2:    │   %4  = (485:506)::Core.Const(485:506)
# From worker 2:    │         (b0 = Base.getindex(θ, %4))
# From worker 2:    │   %6  = (507:990)::Core.Const(507:990)
# From worker 3:    │   %19 = Main.Chain(%16, %17, %18, Main.softmax)::Chain{Tuple{Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, Dense{typeof(identity), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, typeof(softmax)}}
# From worker 2:    │   %7  = Base.getindex(θ, %6)::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:    └──       return %19
# From worker 3:
# From worker 2:    │         (W1 = Main.reshape(%7, 22, 22))
# From worker 2:    │   %9  = (991:1012)::Core.Const(991:1012)
# From worker 2:    │         (b1 = Base.getindex(θ, %9))
# From worker 2:    │   %11 = (1013:1056)::Core.Const(1013:1056)
# From worker 2:    │   %12 = Base.getindex(θ, %11)::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 2:    │         (W2 = Main.reshape(%12, 2, 22))
# From worker 2:    │   %14 = (1057:1058)::Core.Const(1057:1058)
# From worker 2:    │         (b2 = Base.getindex(θ, %14))
# From worker 3:    MethodInstance for feedforward(::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}})
# From worker 3:      from feedforward(θ::AbstractVector) in Main at /Users/456828/Projects/Bayesian-Active-Learning/DataSets/coalminequakes_dataset/Network.jl:8
# From worker 3:    Arguments
# From worker 3:      #self#::Core.Const(feedforward)
# From worker 3:      θ::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:    Locals
# From worker 2:    │   %16 = Main.Dense(W0, b0, Main.mish)::Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}
# From worker 3:      b2::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:      W2::ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}
# From worker 3:      b1::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 2:    │   %17 = Main.Dense(W1, b1, Main.mish)::Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}
# From worker 3:      W1::ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}
# From worker 3:      b0::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:      W0::ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}
# From worker 2:    │   %18 = Main.Dense(W2, b2)::Dense{typeof(identity), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}
# From worker 3:    Body::Chain{Tuple{Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, Dense{typeof(mish), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, Dense{typeof(identity), ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}, Vector{ReverseDiff.TrackedReal{Float32, Float32, ReverseDiff.TrackedArray{Float32, Float32, 2, Matrix{Float32}, Matrix{Float32}}}}}, typeof(softmax)}}



# From worker 2:    MethodInstance for feedforward(::Matrix{Float32}, ::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}})
# From worker 2:      from feedforward(x, theta) in Main at /Users/456828/Projects/Bayesian-Active-Learning/ActiveLearning.jl:96
# From worker 2:    Arguments
# From worker 2:      #self#::Core.Const(feedforward)
# From worker 2:      x::Matrix{Float32}
# From worker 2:      theta::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 2:    Body::Any
# From worker 2:    1 ─ %1 = Main.destructured(theta)::Any
# From worker 2:    │   %2 = (%1)(x)::Any
# From worker 2:    └──      return %2
# From worker 2:
# From worker 3:    MethodInstance for feedforward(::Matrix{Float32}, ::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}})
# From worker 3:      from feedforward(x, theta) in Main at /Users/456828/Projects/Bayesian-Active-Learning/ActiveLearning.jl:96
# From worker 3:    Arguments
# From worker 3:      #self#::Core.Const(feedforward)
# From worker 3:      x::Matrix{Float32}
# From worker 3:      theta::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:    Body::Any
# From worker 3:    1 ─ %1 = Main.destructured(theta)::Any
# From worker 3:    │   %2 = (%1)(x)::Any
# From worker 3:    └──      return %2
# From worker 3:




# From worker 3:    │   %3 = Base.getproperty(re, :length)::Int64
# From worker 2:    1 ─ %1 = Base.getproperty(re, :model)::Chain{Tuple{Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}
# From worker 3:    │   %4 = Optimisers._rebuild(%1, %2, flat, %3)::Chain
# From worker 3:    └──      return %4
# From worker 3:
# From worker 2:    │   %2 = Base.getproperty(re, :offsets)::NamedTuple{(:layers,), Tuple{Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, Tuple{}}}}
# From worker 2:    │   %3 = Base.getproperty(re, :length)::Int64
# From worker 2:    │   %4 = Optimisers._rebuild(%1, %2, flat, %3)::Chain
# From worker 2:    └──      return %4
# From worker 2:
# From worker 3:    MethodInstance for (::Optimisers.Restructure{Chain{Tuple{Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}, NamedTuple{(:layers,), Tuple{Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, Tuple{}}}}})(::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}})
# From worker 3:      from (re::Optimisers.Restructure)(flat::AbstractVector) in Optimisers at /Users/456828/.julia/packages/Optimisers/1x8gl/src/destructure.jl:59
# From worker 3:    Arguments
# From worker 3:      re::Optimisers.Restructure{Chain{Tuple{Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}, NamedTuple{(:layers,), Tuple{Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, Tuple{}}}}}
# From worker 2:    MethodInstance for (::Optimisers.Restructure{Chain{Tuple{Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(mish), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, typeof(softmax)}}, NamedTuple{(:layers,), Tuple{Tuple{NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, NamedTuple{(:weight, :bias, :σ), Tuple{Int64, Int64, Tuple{}}}, Tuple{}}}}})(::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}})
# From worker 3:      flat::ReverseDiff.TrackedArray{Float32, Float32, 1, Vector{Float32}, Vector{Float32}}
# From worker 3:    Body::Chain
# From worker 2:      from (re::Optimisers.Restructure)(flat::AbstractVector) in Optimisers at /Users/456828/.julia/packages/Optimisers/1x8gl/src/destructure.jl:59
# From worker 2:    Arguments
