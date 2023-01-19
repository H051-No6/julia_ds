function linear_regression(x::Matrix, y::Array)
    xt = transpose(x)
    xtx = xt*x
    xty = xt*y
    return inv(xtx)*xty
end

module LinearRegresser
    lmd::Float64
    function fit(x::Matrix, y::Array)
        xt = transpose(x)
        xtx = xt*x
        xty = xt*y
        return inv(xtx)*xty
    end
end