include("DubinsCar.jl")
include("ocp_solvers.jl")
include("trajectories.jl")
using JuMP, HiGHS, Ipopt, OSQP, Plots
using Base.Threads, BenchmarkTools

@kwdef mutable struct SSA_Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int64
    T :: Int64
    Δt :: Float64
    α :: Float64
    ϵ :: Float64
    δ :: Float64
end


L = 200                 # total number of particles
M = 50                  # Monte Carlo Samples
Δt = 0.1                 # time step
T = 20                   # total simulation time
N = Int64(0.05*(T/Δt))           # Time horizon
K = Int64(T/Δt)          # number of time steps
n = 3                    # dimension of state
α = 0.15
ϵ = 0.30
δ = 0.10

ssa_params =  SSA_Params(M,N,L,n,T,Δt,α,ϵ,δ)


## Initalize dubins car objects ##
x0_pursuer = -1.5         # adversary x position
y0_pursuer = 0.0         # adversary y position
θ0_pursuer = pi/2        # adversary orientation
v_pursuer= 1.5           # Fixed pursuer forward velocity
up_max = 0.5             # Bound on input (turn rate)

x0_evader = 0.0          # Initial evader x position
y0_evader = 0.0          # Initial evader y position
θ0_evader = pi/2         # Initial evader orientation
v_evader = 0.75           # Fixed evader forward velocity
ue_max = 1.0             # Bound on input (turn rate)

pursuer_init_state = Vector{Float64}([x0_pursuer, y0_pursuer, θ0_pursuer])
evader_init_state = Vector{Float64}([x0_evader, y0_evader, θ0_evader])


μ = [evader_init_state[1],      # mean initial state
    evader_init_state[2]]
R = 0.75                         # measurement variance
Σ = 0.10                      # process variance

__pursuer = Dubins_Car(f,h,v_pursuer,pursuer_init_state, pursuer_init_state, up_max)
__evader = Dubins_Car(f,h,v_evader,evader_init_state, evader_init_state,ue_max)

# initial state density
Ξ = vcat(Array(gpu_generate_Xi(L, n-1, μ, R)), fill(evader_init_state[3],(L))')


mc_states = Array{Float64}(undef, (n,M,N))

# calculate control sequences for each particle 
u = Array{Float64}(undef,(L,N))
@btime xprime!(ssa_params, Ξ, u, __pursuer, __evader)
