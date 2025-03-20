Base.@kwdef mutable struct ControlAffineDynamics
    f::Function
    g::Function
    A::Matrix{Float64}
    B::Matrix{Float64}
    n::Int64
    m::Int64
end

## DEFINE HCW DYNAMICS  ##
# η = 0.0011  
# # A =  [0 0 1 0;
# #       0 0 0 1;
# #       3*η^2 0 0 2*η;
# #       0 0 -2*η 0]
# A =  [0 0 1 0;
#       0 0 0 1;
#       0 0 0 0;
#       0 0 0 0]
# B =  [0.0 0.0;
#       0.0 0.0;
#       1.0 0.0;
#       0.0 1.0]
# C =  [1.0 0.0 0.0 0.0;
#       0.0 1.0 0.0 0.0]

# # Compute discrete-time A_d and B_d
# #  Time step for discretization
# Δt = 0.1
# Ad = exp(Δt * A)  # Exponential of matrix A
# Bd = Δt * Ad * B  # Euler approximation of integral equation for Bd

# # Gain matrices for nominal feedback controller
# Kp = [1.2 0.0; 
#       0.0 1.2]
# Kd = [1.75 0.0; 
#       0.0 1.75]


# # Noise matrix
# Q =  [0.15 0.0;
#       0.0 0.15;]
# R =  [0.15 0.0;
#       0.0 0.15]


# n = 4 # state dimension
# m = 2 # control dimension

# ## Define control affine system; dx = Ax + Bu
# f(x::Vector{Float64}) = A*x
# g(x::Vector{Float64}) = B
# H(x::Vector{Float64}) = C*x + R*randn(size(C,1))

# ## Define discrete-time affine system; xₖ₊₁ = Axₖ + Buₖ
# fd(x::Vector{Float64}) = Ad*x
# gd(x::Vector{Float64}) = Bd

# ## propagate through discrete time dynamics
# dynamics(Σ::Dynamics,x,u) = Σ.fd(x) + Σ.gd(x)*(u + Σ.Q*randn(Σ.m)) 



 


