using LinearAlgebra
using Expokit
using BenchmarkTools  
using Plots  
using Base.Threads
using JuMP, HiGHS, OSQP

include("SSA_HCW_dynamics.jl")
include("SSA_ControlBarrierFunction.jl")
include("SSA_QP_SafetyFilter.jl")
include("SSA_ParticleFilter.jl")
include("State_Selection_Algorithm.jl")

function gpu_generate_Xi(L :: Int64, n :: Int64, μ::Vector{Float64}, var)
     # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
     μ_x0 = Array(μ)
     Σ_x0 = (var*I)
 
     # randomly sample initial states Ξ following Gaussian density
     Ξ₀ = Array{Float64}(undef,n,L)
     Ξ₀ = μ_x0.+vcat(sqrt(Σ_x0)*randn(2,L), 0.0*randn(2,L))
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
L = 400 # number of particles
M = 135  # number of Monte Carlo samples
T = 10   # total simulation time [seconds]
K = Int64(T/Δt) # total simulation steps
N = Int64(0.10*K)   # time horizon

# intialize parameters
SSA_params = Params(M, N, L, n, T, K)
w = 0.05

# Nominal feedback controller
r_goal = [0.0, 0.0]
kd(x) = -Kp*(x[1:2]-r_goal)-Kd*x[3:4]

# initalize a control affine system
σ = Dynamics(f,g,H,fd,gd,kd,A,B,Q,R,n,m)

# initialize state and control arrays
state = Array{Float64}(undef, (n,K,L))
μ = [3.5, 3.5, 0.0, 0.0]
x0 = Array(gpu_generate_Xi(L,n,μ,w))
x0_true = μ + [w*randn(),w*randn(),0.0,0.0]
state[:,1,:] .= x0

## Define a control barrier function
# x: relative position, [rx, ry]
dmin = 1.25
r0 = [1.6, 1.45]
h(x) = norm(x[1:2]-r0)^2 - dmin^2
α(r) = 1.0*r
cbf = ControlBarrierFunction(h, σ, α)
J(x,u) = h(x)

# Define safety filter with the cbf and nominal controller
k_cbf = QPSafetyFilter(cbf, σ, kd)

# Define boostrap particle filter
pf = Particle_Filter(σ, TimeUpdate, MeasurementUpdate, Resampler, ones(L), x0)
xtrue = Array{Float64}(undef, (n,K))
xtrue[:,1] = x0_true

xstar = Array{Float64}(undef, (n,K))

u_total = Array{Float64}(undef, (m,N,L))

function simulate!(SSA_params, xstar, xtrue, state, Σ::Dynamics, sf::QPSafetyFilter, pf::Particle_Filter,J::Function)
     K = SSA_params.K
     for t = 1:K-1
          Ξ = state[:,t,:]
          # x_star = vec(mean(Ξ, dims=2))
          x_star = state_selection_algorithm(SSA_params, Ξ, x->sf(x), Σ, J)
          u_star = sf(x_star)

          xstar[:,t] = x_star

          xtrue[:,t+1] = dynamics(Σ, xtrue[:,t], u_star)

          y = Σ.H(xtrue[:,t+1])

          propagate_bootstrap_pf(pf,Σ,u_star,Ξ,y)
          state[:,t+1,:] = pf.particles
          println(t)
     end
end

simulate!(SSA_params, xstar, xtrue, state, σ, k_cbf, pf, J)





