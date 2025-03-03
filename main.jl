#push!(LOAD_PATH, "./")
using LinearAlgebra, Revise, ControlSystemsBase, Plots, Statistics
using JuMP, OSQP
using ForwardDiff, DifferentialEquations

include("Mohammad/ParticleFilter.jl")
include("Mohammad/SSA.jl")
include("Mohammad/ControlAffineDynamics.jl")
include("Mohammad/ControlBarrierFunction.jl")
include("Mohammad/QPSafetyFilter.jl")

# Define state-space example
A =  [0 0 1 0;
      0 0 0 1;
      0 0 0 0;
      0 0 0 0]
B =  [0.0 0.0;
      0.0 0.0;
      1.0 0.0;
      0.0 1.0]
C =  [1.0 0.0 0.0 0.0;
      0.0 1.0 0.0 0.0]

# Compute discrete-time A_d and B_d
#  Time step for discretization
Δt = 0.1
Ad = exp(Δt * A)  # Exponential of matrix A
Bd = Δt * Ad * B  # Euler approximation of integral equation for Bd


Σ = ControlAffineDynamics(x -> A*x, x -> B, A, B, 4, 2)

# Nominal feedback controller
# Gain matrices for nominal feedback controller
Kp = [1.2 0.0; 
      0.0 1.2]
Kd = [1.75 0.0; 
      0.0 1.75]
r_goal = [0.0, 0.0]
kd(x) = -Kp*(x[1:2]-r_goal)-Kd*x[3:4]

# Noise matrix
W = Diagonal([0.001, 0.001, 0.005, 0.005])
V = Diagonal([0.05, 0.05])

f(x::Vector{Float64}, u::Vector{Float64}) = Ad * x + Bd * u + sqrt(W) * randn(size(W,1))

h(x::Vector{Float64}) = C * x + sqrt(V) * randn(size(V,1))

sys = ss(Ad,Bd,C,0,Δt)

Q = I
R = 2I

K_LQR = -lqr(sys, Q, R)
println("LQR control gain ", K_LQR)
controller(x::Vector{Float64}) = K_LQR * x

## Define a control barrier function
dmin = 1.25
r0 = [1.6, 1.45]
h_cbf(x) = norm(x[1:2]-r0)^2 - dmin^2
kappa(r) = 1.0*r
cbf = ControlBarrierFunction(h_cbf, Σ, kappa)

# Define safety filter with the cbf and nominal controller
k_cbf = QPSafetyFilter(cbf, Σ, kd)


L = 300
initial_particles = [3.0; 3.0; 0.0; 0.0] .+ sqrt(Diagonal([1, 1, 0, 0])) * randn(size(W,1), L)
initial_likelihoods = ones(L) / L
pf = ParticleFilter(f, h, W, V, initial_particles, initial_likelihoods)

cost(x::Vector{Float64}, u::Vector{Float64}) = x' * Q * x + u' * R * u
constraint_violation(x::Vector{Float64}) = h_cbf(x) < 0
N = 5
M = 100
α = 0.15
ssa = SSA(pf, x->k_cbf(x), N, M, cost, constraint_violation, α)

function simulate_SSA(x_true0, T)
      X_true_queue = [x_true0]
      X_star_queue = []
      particle_queue = []
      for t = 1:T
            println("Time step: ", t)
            x_prime0, α_t_achieved, cost_t_achieved = SSA_sample_averages(ssa)
            x_prime_optimal = SSA_select(ssa, x_prime0, α_t_achieved, cost_t_achieved)

            # x_prime_optimal = vec(mean(ssa.PF.particles, dims=2))
            u = ssa.K₀(x_prime_optimal)
            y = ssa.PF.h(x_true0)
            propagate_PF!(ssa.PF, u, y)
            x_true0 = ssa.PF.f(x_true0, u)
            push!(particle_queue, copy(ssa.PF.particles))
            push!(X_true_queue, x_true0)
            push!(X_star_queue, x_prime_optimal)
      end
      return  X_star_queue, particle_queue
end

T = 70
x_true0 = [3.0; 3.0; 0.0; 0.0] .+ sqrt(Diagonal([1, 1, 0, 0])) * randn(size(W,1))

x_star_queue, particle_queue = simulate_SSA(x_true0, T)