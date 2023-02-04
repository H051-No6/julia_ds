using LinearAlgebra

function fit(X, y, settings)
    error("error! $settings is not supported type")
end

Base.@kwdef struct LinearRegression
    add_const::Bool = false
end

function LinearRegression()
    return LinearRegression(false)
end

function fit(X::Matrix, y::Array, model::LinearRegression)
    if model.add_const
        X = cat(X, ones(size(X)[1]), dims=2)
    end
    xt = transpose(X)
    xtx = xt*X
    xty = xt*y
    inv(xtx)*xty
end

Base.@kwdef struct RidgeRegression
    lambda::Float64 = 0.
    add_const::Bool = false
end

function RidgeRegression(lambda=0., add_const=false)
    return RidgeRegression(lambda, add_const)
end

function fit(X::Matrix, y::Array, model::RidgeRegression)
    xt = transpose(X)
    xtx = xt*X
    xty = xt*y
    inv(xtx+model.lambda*I)*xty
end