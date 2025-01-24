# Discrete-time Dubins Car Dynamics
@kwdef mutable struct Dubins_Car
    f::Function
    h::Function
    v::Float64
    x_true::Vector{Float64}
    x_measured::Vector{Float64}
    u_max::Float64
end

function f(x::Vector{Float64},u::Float64,w::Vector{Float64},v::Float64,δt::Float64)
    θ = x[3]
    xk = x + δt*[v*cos(θ) + w[1]; v*sin(θ) + w[2]; u]
    return xk
end

function h(x::Vector{Float64},v)
    return y = x[1:2]+v
end