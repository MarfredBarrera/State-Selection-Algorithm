#push!(LOAD_PATH, "./")
using LinearAlgebra, Revise, ControlSystemsBase, Plots, Statistics
using JuMP, OSQP
using ForwardDiff, DifferentialEquations

include("SSA_CBFv2/ParticleFilter.jl")
include("SSA_CBFv2/SSA.jl")
include("SSA_CBFv2/ControlAffineDynamics.jl")
include("SSA_CBFv2/ControlBarrierFunction.jl")
include("SSA_CBFv2/QPSafetyFilter.jl")

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

# Control Affine Dynamics for use in constructing Control Barrier Function
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
W = Diagonal([0.001, 0.001, 0.0025, 0.0025])
V = Diagonal([0.15, 0.15])

# state transition
f(x::Vector{Float64}, u::Vector{Float64}) = Ad * x + Bd * u + sqrt(W) * randn(size(W,1))
# f(x::Vector{Float64}, u::Vector{Float64}) = Ad * x + Bd * u
# measurement
h(x::Vector{Float64}) = C * x + sqrt(V) * randn(size(V,1))
# h(x::Vector{Float64}) = C * x

sys = ss(Ad,Bd,C,0,Δt)

Q = I
R = 2*I

K_LQR = -lqr(sys, Q, R)
# println("LQR control gain ", K_LQR)
controller(x::Vector{Float64}) = K_LQR * x

# Construct a control barrier function
dmin = 1.25
r0 = [1.85, 1.55]
h_cbf(x) = norm(x[1:2]-r0)^2 - dmin^2
kappa(r) = 1.0*r
cbf = ControlBarrierFunction(h_cbf, Σ, kappa)

# Define safety filter with the cbf and nominal controller
k_cbf = QPSafetyFilter(cbf, Σ, kd)

# Define initial state density
L = 300
var0 = Diagonal([0.15, 0.15, 0, 0])
μ0 = [2.65; 3.00; 0.0; 0.0;]
initial_particles = μ0 .+ sqrt(var0) * randn(size(W,1), L)

# Define particle filter
initial_likelihoods = ones(L) / L
pf = ParticleFilter(f, h, W, V, initial_particles, initial_likelihoods)

cost(x::Vector{Float64}, u::Vector{Float64}) = x' * Q * x + u' * R * u
constraint_violation(x::Vector{Float64}) = h_cbf(x) < 0
N = 5
M = 150
α = 0.15
ssa = SSA(pf, x->k_cbf(x), N, M, cost, constraint_violation, α)

function simulate_SSA(x_true0, T, ssa, RUN_SSA)
      X_true_queue = [x_true0]
      X_star_queue = []
      particle_queue = []
      violation_queue = []
      cost_queue = []
      for t = 1:T


            if(RUN_SSA)
                  x_prime0, α_t_achieved, cost_t_achieved = SSA_sample_averages(ssa)
                  x_prime_optimal = SSA_select(ssa, x_prime0, α_t_achieved, cost_t_achieved)
            else
                  x_prime_optimal = vec(mean(ssa.PF.particles, dims=2))
            end

            u = ssa.K₀(x_prime_optimal)
            y = ssa.PF.h(x_true0)
            propagate_PF!(ssa.PF, u, y)
            x_true0 = ssa.PF.f(x_true0, u)
            push!(particle_queue, copy(ssa.PF.particles))
            push!(X_true_queue, x_true0)
            push!(X_star_queue, x_prime_optimal)
            push!(violation_queue, check_α(ssa.PF.particles))
            push!(cost_queue, sum(ssa.running_cost(ssa.PF.particles[:,i],u) for i in size(ssa.PF.particles,2)))

            print("Time step: ", t)
            print(", Constraint violation: ", check_α(ssa.PF.particles), "\n")
      end
      return  X_star_queue, particle_queue, violation_queue, cost_queue
end

function check_α(Xi::Matrix{Float64})
      alpha_sum = 0.0
      for i = 1:size(Xi, 2)
            alpha_sum += constraint_violation(Xi[:,i])
      end
      return alpha_sum / size(Xi, 2)
end

T = 70
x_true0 = μ0 .+ sqrt(var0) * randn(size(W,1))

run_ssa = true

# run a simulation with SSA
x_star_queue, particle_queue, violation_queue, cost_queue = simulate_SSA(x_true0, T, ssa, run_ssa)


# reset pf and ssa
pf = ParticleFilter(f, h, W, V, initial_particles, initial_likelihoods)
ssa = SSA(pf, x->k_cbf(x), N, M, cost, constraint_violation, α)

# run a simulation without SSA
x_star_queue_noSSA, particle_queue_noSSA, violation_queue_noSSA, cost_queue_noSSA = simulate_SSA(x_true0, T, ssa, !run_ssa)
