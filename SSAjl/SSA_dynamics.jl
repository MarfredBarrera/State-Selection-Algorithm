
Base.@kwdef mutable struct Model
    f::Function
    h::Function
    u::Function
    Q::Matrix{Float64}
    R::Float64
end

## motion model dynamics
function f(x,u,w)
    xk = Vector{Float64}(undef, 2)
    
    xk[1] = 0.9*x[1] + 0.2*x[2] + w[1]
    xk[2] = -0.15*x[1] + 0.9*x[2] + 0.05*x[1]*x[2] + u + w[2]

    return xk
end

## measurement model dynamics
function h(x,v)
    return x[1,:] .+ v
end

## controller
function u(x)
    return -0.05*x[1]*x[2]
end