using LinearAlgebra
using Expokit
using BenchmarkTools  
using Plots  
using CUDA, Base.Threads
using JuMP, HiGHS, OSQP

include("SSA_HCW_dynamics.jl")
include("SSA_ControlBarrierFunction.jl")
include("SSA_QP_SafetyFilter.jl")
include("SSA_ParticleFilter.jl")

function gpu_generate_Xi(L :: Int64, n :: Int64, μ::Vector{Float64}, var)
     # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
     μ_x0 = CuArray(μ)
     Σ_x0 = (var*I)
 
     # randomly sample initial states Ξ following Gaussian density
     Ξ₀ = CuArray{Float64}(undef,n,L)
     Ξ₀ = μ_x0.+vcat(sqrt(Σ_x0)*CUDA.randn(2,L), 0.0*CUDA.randn(2,L))
     return (Ξ₀)
 end
 
 Base.@kwdef struct Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int64
    T :: Int64
    K :: Int64
end

## DEFINE SSA PARAMETERS ##
# Density and sampling parameters
L = 250 # number of particles
M = 100  # number of Monte Carlo samples
T = 30   # total simulation time [seconds]
K = Int64(T/Δt) # total simulation steps
N = K   # time horizon

# intialize parameters
SSA_params = Params(M, N, L, n, T, K)
w = 0.15

# Nominal feedback controller
r_goal = [0.0, 0.0]
kd(x) = -Kp*(x[1:2]-r_goal)-Kd*x[3:4]

# initalize a control affine system
σ = Dynamics(f,g,H,fd,gd,kd,A,B,Q,R,n,m)

# initialize state and control arrays
state = Array{Float64}(undef, (n,N,L))
u_qp = Array{Float64}(undef, (m,N,L))
μ = [3.0,3.0, 0.0, 0.0]
x0 = Array(gpu_generate_Xi(L,n,μ,w))
x0_true = μ + [w*randn(),w*randn(),0.0,0.0]
state[:,1,:] .= x0

## Define a control barrier function
# x: relative position, [rx, ry]
dmin = 1.0
r0 = [0.0, 0.0]
h(x) = norm(x[1:2]-r0)^2 - dmin^2
α(r) = 1.0*r
cbf = ControlBarrierFunction(h, σ, α)

# Define safety filter with the cbf and nominal controller
k_cbf = QPSafetyFilter(cbf, σ, kd)


pf = Particle_Filter(σ, TimeUpdate, MeasurementUpdate, Resampler, ones(L), x0)
xtrue = Array{Float64}(undef, (n,K))
xtrue[:,1] = x0_true

function simulate!(SSA_params, xtrue, state, Σ::Dynamics, sf::QPSafetyFilter, pf::Particle_Filter)
     K = SSA_params.K
     t = 1
     for t = 1:K-1
          Ξ = state[:,t,:]
          x_star = vec(mean(pf.particles, dims=2))
          u_star = sf(x_star)
          xtrue[:,t+1] = dynamics(Σ, xtrue[:,t], u_star)

          y = σ.H(xtrue[:,t+1])

          propagate_bootstrap_pf(pf,Σ,u_star,Ξ,y)
          state[:,t+1,:] = pf.particles
     end
end





simulate!(SSA_params, xtrue, state, σ, k_cbf, pf)
# simulate!(SSA_params, state, σ, k_cbf, pf)






