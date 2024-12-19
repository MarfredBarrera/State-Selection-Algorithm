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
M = 200
N = 6
L = 500
n = 2
T = 20

# state density mean and variance
μ = [7.5;-7.5]
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
y2_lowerlim = -7

 

# store parameters in struct
SSA_params = Params(M, N, L, n, T)
# SSA_gauss  = Gaussians(μ, Σ, ω, v)
SSA_limits = Limits(Ulim, α, ϵ, δ, 
    x1_upperlim, x1_lowerlim, 
    y1_upperlim, y1_lowerlim,
    x2_upperlim, x2_lowerlim, 
    y2_upperlim, y2_lowerlim)


RUN_SSA = false
RUN_SIMULATIONS = false

# simulation parameters
RUN_SSA ? RUN_CM = false : RUN_CM = true

RUN_SIMULATIONS ? RUN_PLOTS = false : RUN_PLOTS = true
ANIMATE = false

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

        if(RUN_CM)
            gif(anim, "cm.gif",fps=10)
        elseif(RUN_SSA)
            gif(anim, "ssa.gif",fps=10)
        end
    end

    for i = 2:T
        animate_frame(i)
    end

    if(RUN_CM)
        savefig("cm_plot.png")
    elseif(RUN_SSA)
        savefig("ssa_plot.png")
    end

end



