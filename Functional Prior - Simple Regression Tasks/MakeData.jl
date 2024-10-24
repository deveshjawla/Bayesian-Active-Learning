function f(x, name::String)
    if name == "Polynomial"
        f = x^4 + 2 * x^3 - 3 * x^2 + x
    elseif name == "Cosine"
        f = cos(x)#sin(π * sin(x))
    elseif name == "sin(pisin)"
        f = sin(π * sin(x))
    elseif name == "Exponential"
        f = exp(x)
    elseif name == "Logarithmic"
        f = log(x)
    end
    return f
end

function make_data(function_name::String)

    if function_name == "Polynomial"
        # f(x) = x^4 + 2 * x^3 - 3 * x^2 + x 
        xs1 = collect(Float32, -3.36:0.02:-2.9)# .
        xs2 = collect(Float32, -1:0.02:1.8)# .
        xs = vcat(xs1, xs2)
        Xline = collect(Float32, -3.6:0.01:2.3)
        ys = map(x -> f(x, function_name) + randn(), xs)
    elseif function_name == "Cosine" || function_name == "sin(pisin)"
        # f(x) = cos(x) 
        xs1 = collect(Float32, -7:0.05:-1)# .
        xs2 = collect(Float32, 3:0.05:5)# .
        xs = vcat(xs1, xs2)
        Xline = collect(Float32, -15:0.01:15)
        ys = map(x -> f(x, function_name) + randn(), xs)
    elseif function_name == "Exponential"
        # f(x) = exp(x) 
        xs1 = collect(Float32, -1:0.01:1)# .
        xs2 = collect(Float32, 2:0.01:4.5)# .
        xs = vcat(xs1, xs2)
        Xline = collect(Float32, -2:0.01:5)
        ys = map(x -> f(x, function_name) + randn(), xs)

    elseif function_name == "Logarithmic"
        # f(x) = log(x) 
        xs1 = collect(Float32, 0.01:0.005:1)# .
        xs2 = collect(Float32, 2:0.01:3)# .
        xs = vcat(xs1, xs2)
        Xline = collect(Float32, 0:0.005:6)
        ys = map(x -> f(x, function_name) + randn(), xs)

    end
    return xs, ys, Xline
end
