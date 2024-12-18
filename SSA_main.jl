include("SSA_utils.jl")
include("SSA_particle_filter.jl")
include("SSA_dynamics.jl")
include("SSA_kernels.jl")
include("SSA_plotting.jl")
using Plots

Base.@kwdef struct Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int64
    T :: Int64
end

Base.@kwdef struct Gaussians
    # state density with mean and variance
    μ :: Float64
    Σ :: Float64

    # process noise ωₖ and scalar measurement noise vₖ
    ω :: Float64
    v :: Float64
end

Base.@kwdef struct Limits
    Ulim :: Float64
    α :: Float32
    ϵ :: Float32
    δ :: Float64

    x1_upperlim :: Float64
    x1_lowerlim :: Float64
    y1_upperlim :: Float64
    y1_lowerlim :: Float64

    x2_upperlim :: Float64
    x2_lowerlim :: Float64
    y2_upperlim :: Float64
    y2_lowerlim :: Float64
end

# intialize parameters
M = 275
N = 6
L = 300
n = 2
T = 10

# state density mean and variance
μ = [3;-8]
Σ = 0.5^2

#  process noise variance ωₖ and scalar measurement noise variance vₖ
ω = 0.3^2
v = 0.3^2

Ulim = 3
α = 0.20
ϵ = 0.40
δ = 0.01

# state constraints
x1_upperlim = 5
x1_lowerlim = 3
y1_upperlim = 2
y1_lowerlim = -4

x2_upperlim = 5
x2_lowerlim = -2
y2_upperlim = -4
y2_lowerlim = -5

 

# store parameters in struct
SSA_params = Params(M, N, L, n, T)
# SSA_gauss  = Gaussians(μ, Σ, ω, v)
SSA_limits = Limits(Ulim, α, ϵ, δ, 
    x1_upperlim, x1_lowerlim, 
    y1_upperlim, y1_lowerlim,
    x2_upperlim, x2_lowerlim, 
    y2_upperlim, y2_lowerlim)

# simulation parameters

RUN_SIMULATIONS = true
RUN_PLOTS = false
MARKER_SIZE = 1
ANIMATE = false
state_constraints = [
    rectangle_from_coords(x1_lowerlim,y1_lowerlim,x1_upperlim,y1_upperlim)
    rectangle_from_coords(x2_lowerlim,y2_lowerlim,x2_upperlim,y2_upperlim)]

# simulation state machine
global sim_data
global x_candidate
global violation_rate

if(RUN_SIMULATIONS)
    global (x_candidate, sim_data, violation_rate) = run_simulation(T)

elseif(RUN_PLOTS)

    plot(sim_data[1,:,1],sim_data[2,:,1],seriestype=:scatter,label=false,ms=MARKER_SIZE)
    scatter
    scatter!((x_candidate[1,1], x_candidate[2,1]),
    label = false,
    mc =:red,
    z_order=:front)

    if(ANIMATE)
        anim = @animate for i = 2:T
            animate_frame(i)
        end
        gif(anim, "ssa.gif",fps=10)
    end

    for i = 2:T
        animate_frame(i)
    end
    savefig("plot.png")
end

function run_conditional_mean_sim(T)

    sim_data = fill(0.0f0, (n,L,T))
    x_true = fill(0.0f0, (n,T))
    violation_rate = fill(0.0f0,T)
    x_mean = fill(0.0f0, (n,T))

    # initialize dynamics
    Q = Matrix{Float64}(I, 2, 2)
    R = v
    dynamics = Model(f,h,u,Q,R)
    x_true = Array{Float64}(undef, n, T+1)

    Ξ = gpu_generate_Xi(L,n,μ)

    # initialize particle filter
    likelihoods = Vector(fill(1,(L)))
    pf = Particle_Filter(dynamics, TimeUpdate, MeasurementUpdate!, Resampler, likelihoods, Array(Ξ))
    x_true[:,1] = μ.+ sqrt(Σ)*randn(2)

    for t = 1:T
        pf.particles = Array(Ξ)

    end


end


