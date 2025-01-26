using DifferentialEquations
using ForwardDiff

Base.@kwdef mutable struct Dynamics
    f::Function
    g::Function
    fd::Function
    gd::Function
    kd::Function
    A::Matrix{Float64}
    B::Matrix{Float64}
    Q::Matrix{Float64}
    R::Float64
    n::Int64
    m::Int64
end

## DEFINE HCW DYNAMICS PARAMETERS ##
η = 0.0011  
A =  [0 0 1 0;
      0 0 0 1;
      3*η^2 0 0 2*η;
      0 0 -2*η 0]
# A =  [0.0 0.0;
#       0.0 0.0]

B =  [0.0 0.0;
      0.0 0.0;
      1.0 0.0;
      0.0 1.0]
# B =  [1.0 0.0; 
#       0.0 1.0] 

# Compute discrete-time A_d and B_d
#  Time step for discretization
Δt = 0.01   
Ad = exp(Δt * A)  # Exponential of matrix A
Bd = Δt * Ad * B  # Euler approximation of integral equation for Bd

# Gain matrices for nominal feedback controller
Kp = [1.2 0.0; 
      0.0 1.2]
Kd = [1.75 0.0; 
      0.0 1.75]


# Noise matrix
# Q =  [0.0 0.0 0.0 0.0;
#       0.0 0.0 0.0 0.0;
#       0.0 0.0 0.0 0.0;
#       0.0 0.0 0.0 0.0;]
Q =  [2.0 0.0;
      0.0 2.0;]

R = 0.0


n = 4 # state dimension
m = 2 # control dimension

## Define control affine system; dx = Ax + Bu
f(x::Vector{Float64}) = A*x
g(x::Vector{Float64}) = B

## Define discrete-time affine system; xₖ = Ax + Bu
fd(x::Vector{Float64}) = Ad*x
gd(x::Vector{Float64}) = Bd

## propagate through discrete time dynamics
dynamics(Σ::Dynamics,x,u) = Σ.fd(x) + Σ.gd(x)*(u + Σ.Q*randn(Σ.m))

## define HCW dynamics
# HCW = Dynamics(f,g,fd,gd,kd,Q,R,n,m)
 



 


