using LinearAlgebra
using Expokit
using BenchmarkTools  
using Plots  
using CUDA, Base.Threads
using JuMP, HiGHS, OSQP

include("SSA_HCW_dynamics.jl")
include("SSA_ControlBarrierFunction.jl")
include("SSA_QP_SafetyFilter.jl")

function gpu_generate_Xi(L :: Int64, n :: Int64, μ::Vector{Float64}, var)
     # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
     μ_x0 = CuArray(μ)
     Σ_x0 = (var*I)
 
     # randomly sample initial states Ξ following Gaussian density
     Ξ₀ = CuArray{Float64}(undef,n,L)
     Ξ₀ = μ_x0.+sqrt(Σ_x0)*CUDA.randn(n,L)
     return (Ξ₀)
 end
 
 Base.@kwdef struct Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int64
    T :: Int64
end

## DEFINE SSA PARAMETERS ##
# Density and sampling parameters
L = 1 # number of particles
M = 100  # number of Monte Carlo samples
T = 30   # simulation total time steps
K = Int64(30/Δt)
N = 500   # time horizon

# intialize parameters
SSA_params = Params(M, N, L, n, T)
w = 0.15

# Nominal feedback controller
r_goal = [0.0, 0.0]
kd(x) = -Kp*(x-r_goal)

# initalize a control affine system
σ = Dynamics(f,g,fd,gd,kd,Q,R,n,m)

# initialize state and control arrays
state = Array{Float64}(undef, (n,N,L))
u_qp = Array{Float64}(undef, (m,N,L))
μ = [4.0,4.0]
x0 = Array(gpu_generate_Xi(L,n,μ,w))
state[:,1,:] .= μ

## Define a control barrier function
# x: relative position, [rx, ry]
dmin = 1.0
r0 = [1.5, 1.5]
h(x) = norm(x[1:2]-r0)^2 - dmin^2
α(r) = 1.0*r
cbf = ControlBarrierFunction(h, σ, α)

# Define safety filter with the cbf and nominal controller
k_cbf = QPSafetyFilter(cbf, σ, kd)

function simulate!(SSA_params, state, Σ, cbf, sf)
     N = SSA_params.N
     L = SSA_params.L

     for i = 1:L
          for t = 1:N-1
               x = state[:,t,i]
               u_safe = sf(x)
               state[:,t+1,i] = dynamics(Σ, x, u_safe)
          end
     end
end

function circleShape(h,k,r)
     θ = LinRange(0,2*pi,500)
     h .+ r*sin.(θ), k.+r*cos.(θ)
end

simulate!(SSA_params, state, σ, cbf, k_cbf)
plot(circleShape(r_goal[1],r_goal[2],0.1), seriestype = [:shape,], lw=0.5,
c =:green, linecolor =:black,
legend = false, aspect_ratio =1)
plot!(circleShape(r0[1],r0[2],dmin), seriestype = [:shape,], lw=0.5,
     c =:red, linecolor =:black,
     legend = false, aspect_ratio =1)

plot!(state[1,:,1], state[2,:,1], seriestype=:scatter, ms = 1.0)



