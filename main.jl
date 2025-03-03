#push!(LOAD_PATH, "./")
using LinearAlgebra, Revise, ControlSystemsBase, Plots

include("Mohammad/ParticleFilter.jl")
include("Mohammad/SSA.jl")
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

# Gain matrices for nominal feedback controller


# Noise matrix
W = Diagonal([0.15, 0.15, 0.15, 0.15])
V = Diagonal([0.15, 0.15])

f(x::Vector{Float64}, u::Vector{Float64}) = Ad * x + Bd * u + sqrt(W) * randn(size(W,1))

h(x::Vector{Float64}) = C * x + sqrt(V) * randn(size(V,1))

sys = ss(Ad,Bd,C,0,Δt)

Q = I
R = 2I

K_LQR = -lqr(sys, Q, R)
println("LQR control gain ", K_LQR)
controller(x::Vector{Float64}) = K_LQR * x

L = 200
initial_particles = sqrt(I) * randn(size(W,1), L)
initial_likelihoods = ones(L) / L
pf = ParticleFilter(f, h, W, V, initial_particles, initial_likelihoods)

cost(x::Vector{Float64}, u::Vector{Float64}) = x' * Q * x + u' * R * u
constraint_violation(x::Vector{Float64}) = 0.0
N = 5
M = 100
α = 0.15
ssa = SSA(pf, controller, N, M, cost, constraint_violation, α)



function simulate_SSA(x_true0, T)
      X_true_queue = [x_true0]
      particle_queue = []
      for t = 1:T
            println("Time step: ", t)
            x_prime0, α_t_achieved, cost_t_achieved = SSA_sample_averages(ssa)
            x_prime_optimal = SSA_select(ssa, x_prime0, α_t_achieved, cost_t_achieved)
            u = ssa.K₀(x_prime_optimal)
            y = ssa.PF.h(x_true0)
            propagate_PF!(ssa.PF, u, y)
            x_true0 = ssa.PF.f(x_true0, u)
            push!(particle_queue, copy(ssa.PF.particles))
            push!(X_true_queue, x_true0)
      end
      return  X_true_queue, particle_queue
end

T = 2
x_true0 = zeros(size(W,1))

simulate_SSA(x_true0, T)
