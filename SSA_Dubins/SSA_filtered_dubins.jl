using JuMP, HiGHS, Ipopt
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


@kwdef mutable struct Parameters
    N::Int64
    Δt::Float64
    T::Int64
end


# Parameters
T = 20                   # Total sim time [seconds]
Δt = 0.5                 # Time step
N = 0.2*(T/Δt)           # Time horizon

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



sim_params = Parameters(N,Δt,T)

# pursuer and evader objects
__pursuer = Dubins_Car(f,h,v_pursuer,pursuer_init_state, pursuer_init_state, up_max)
__evader = Dubins_Car(f,h,v_evader,evader_init_state, evader_init_state,ue_max)

evader_model = Model(Ipopt.Optimizer)
pursuer_model = Model(Ipopt.Optimizer)

K = Int64(sim_params.T / sim_params.Δt)
xₚ = fill(0.0f0, (3,K))
xₑ = fill(0.0f0, (3,K))

xₚ[:,1] = pursuer_init_state
xₑ[:,1] = evader_init_state
