#=
10 December 2020

Utility functions for the random forest
=#


using DataFrames

check_random_state(seed::Int) = MersenneTwister(seed)
check_random_state(rng::AbstractRNG) = rng

# https://en.wikipedia.org/wiki/Histogram
# estimate number of bins for histograms
Sturges_bins(x) = ceil(Int, log2(length(x))) + 1
function Freedman_Diaconis_bins(x::Vector)
    n = length(x)
    x_sort = sort(x)
    idx_Q1 = floor(Int, 0.25*length(x_sort))
    idx_Q3 = ceil(Int, 0.75*length(x_sort))
    IQR = x[idx_Q3] - x[idx_Q1]
    nbins = ceil(Int, 2*IQR/(n^(1/3)) )
    return nbins
end

# split data into training and test data
function split_data(X::DataFrame, Y::DataFrame; test_size=0.1, rng=Random.GLOBAL_RNG)
    random_state = check_random_state(rng)

    n_samples = nrow(X)
    n_train = n_samples - round(Int, test_size * n_samples)

    inds = randperm(random_state, n_samples)
    X = X[inds, :]
    Y = Y[inds, :]

    X_train = X[1:n_train, :]
    Y_train = Y[1:n_train, :]
    X_test  = X[n_train+1:end, :]
    Y_test  = Y[n_train+1:end, :]

    return  X_train, Y_train, X_test, Y_test
end

"""
confusion_matrix(y_actual::Vector{T}, y_pred::Vector{T}) => Matrix{T}

Returns a confusion matrix where the rows are the actual classes, and the columns are the predicted classes
"""
function confusion_matrix(y_actual::Vector{T}, y_pred::Vector{T}) where T
    if length(y_actual) != length(y_pred)
        throw(DimensionMismatch("y_actual length is not the same as y_pred length"))
    end
    n = max(maximum(y_actual), maximum(y_pred))
    C = zeros(Int, n, n)
    for label_actual in 1:n
        idxs_true = y_actual .== label_actual
        for label_pred in 1:n
            C[label_actual, label_pred] = count(y_pred[idxs_true] .== label_pred)
        end
    end
    return C
end

function calc_f1_score(y_actual::Vector{T}, y_pred::Vector{T}) where T
    C = confusion_matrix(y_actual, y_pred)
    return calc_f1_score(C)
end

"""
    calc_f1_score(C::Matrix)

Calculate the recall, precision, f1 on a 2x2 confusion matrix.

\n recall    = true positive / actual positive
\n precision = true positive / predicted positive
\n f1        = 2 * recall *  precision/(recall + precision)

Warning: precision is a core base function, so do not use it as a variable name
"""
function calc_f1_score(C::Matrix)
    recall    = C[2, 2] / (C[2, 1] + C[2, 2]) #true positive/actual positive
    precis = C[2, 2] / (C[1, 2] + C[2, 2]) #true positive/predicted positive
    if ((recall == 0) || (precis == 0))
        f1 = 0
    else
        f1 = 2 * recall*precis/(recall + precis) # = 2/((1/recall)+(1/precision))
    end
    return recall, precis, f1
end

"""
    get_methods_with(mod::Module, type::Type)

Custom implementation of methodswith.
Retrieves all methods which use this type.
"""
function get_methods_with(mod::Module, type::Type)
    mod_name = string(Module)
    identifier = string(type)
    methods_list = Method[]
    for name in names(mod)
        for method in methods(getproperty(mod, name))
            if method.module != mod
                continue
            end
            sig = string(method.sig)
            sig_parts = split(sig, ['{', '}', ','])
            if identifier in sig_parts
                push!(methods_list, method)
            end
        end
    end
    return methods_list
end

macro timeout(expr, seconds=-1, cb=(tsk) -> Base.throwto(tsk, InterruptException()))
    quote
        tsk = @task $expr
        schedule(tsk)

        if $seconds > -1
            Timer((timer) -> $cb(tsk), $seconds)
        end

        return fetch(tsk)
    end
end

macro timeout(seconds, expr, err_expr=:(nothing))
    esc(quote
        tsk__ = @task $expr
        schedule(tsk__)
        start_time__ = time()
        curt__ = time()
        Base.Timer(0.001, interval=0.001) do timer__
            if tsk__ === nothing || istaskdone(tsk__)
                close(timer__)
            else
                curt__ = time()
                if curt__ - start_time__ > $seconds
                    Base.throwto(tsk__, InterruptException())
                end
            end
        end
        try
            fetch(tsk__)
        catch err__
            if err__.task.exception isa InterruptException
                RemoteHPC.log_error(RemoteHPC.StallException(err__))
                $err_expr
            else
                rethrow(err__.task.exception)
            end
        end
    end)
end

@timeout (sleep(3); println("done"), 2)

@timeout (sleep(3); println("done")) 4

macro timeout(seconds, expr, fail)
    quote
        tsk = @task $esc(expr)
        schedule(tsk)
        Timer($(esc(seconds))) do timer
            istaskdone(tsk) || Base.throwto(tsk, InterruptException())
        end
        try
            fetch(tsk)
        catch _
            $(esc(fail))
        end
    end
end


# Parse Dictionary from a String
cl_dist_string = m[2, 2] * "," * m[2, 3]
class_dict = eval(Meta.parse(cl_dist_string))


# using MultivariateStats

# M = fit(PCA, train_x', maxoutdim=150)
# train_x_transformed = MultivariateStats.transform(M, train_x')

# # M = fit(PCA, test_x', maxoutdim = 150)
# test_x_transformed = MultivariateStats.transform(M, test_x')

# train_x = train_x_transformed'
# test_x = test_x_transformed'