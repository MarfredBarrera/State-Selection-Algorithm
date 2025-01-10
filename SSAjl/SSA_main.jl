include("SSA_utils.jl")
include("SSA_particle_filter.jl")
include("SSA_dynamics.jl")
include("SSA_kernels.jl")
include("SSA_plotting.jl")
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

## Choose:
# 1) Either State Selection Algorithm OR Conditional Mean Selection
RUN_SSA = true
# 2) Either compute simulation data OR plot simulation data
COMPUTE_SIM_DATA = false

# simulation parameters
RUN_SSA ? RUN_CM = false : RUN_CM = true


COMPUTE_SIM_DATA ? RUN_PLOTS = false : RUN_PLOTS = true
ANIMATE = false

Ulim = 3
α = 0.10
ϵ = 0.30
δ = 0.01
L = 1000

M = 300

# intialize parameters
N = 6
n = 2
T = 20

# state density mean and variance
μ = [7.75;-7.75]
Σ = 0.5^2

#  process noise variance ωₖ and scalar measurement noise variance vₖ
ω = 0.3^2
v = 0.3^2

# state constraints
x1_upperlim = 5
x1_lowerlim = 3
y1_upperlim = 2
y1_lowerlim = -4

x2_upperlim = 5
x2_lowerlim = -2
y2_upperlim = -4
y2_lowerlim = -7


# initialize dynamics
Q = Matrix{Float64}(I, 2, 2)
R = v
dynamics = Model(f,h,u,Q,R)

# initialize particle filter
likelihoods = Vector(fill(1,(L)))
pf = Particle_Filter(dynamics, TimeUpdate, MeasurementUpdate!, Resampler, likelihoods, Array{Float64}(undef))

# store parameters in struct
SSA_params = Params(M, N, L, n, T)
SSA_limits = Limits(Ulim, α, ϵ, δ, 
    x1_upperlim, x1_lowerlim, 
    y1_upperlim, y1_lowerlim,
    x2_upperlim, x2_lowerlim, 
    y2_upperlim, y2_lowerlim)

# simulation state machine
global sim_data
global x_candidate
global violation_rate

global ssa_data
global cm_data
if(COMPUTE_SIM_DATA)
    global (x_candidate, sim_data, violation_rate) = run_simulation(T)
    if(RUN_SSA)
        ssa_data = violation_rate
        display_name = "State Selection Algorithm"
    elseif(RUN_CM)
        cm_data = violation_rate
        display_name = "Conditional Mean"
    end
    
    println("Simulation Done!\n\n")
    println("Results:")
    println("State Estimation Type: ", "[", display_name, "]")
    println("Total Particles: ", L)
    println("Monte Carlo Samples: ", M)
    println("Max Violation Rate: ",  maximum(violation_rate))

elseif(RUN_PLOTS)

    # store rectangle coordinatse
    state_constraints = [
    rectangle_from_coords(x1_lowerlim,y1_lowerlim,x1_upperlim,y1_upperlim)
    rectangle_from_coords(x2_lowerlim,y2_lowerlim,x2_upperlim,y2_upperlim)]

    # plot initial particle density
    plot(sim_data[1,:,1],sim_data[2,:,1],seriestype=:scatter,label=false,ms=MARKER_SIZE)
    scatter
    scatter!((x_candidate[1,1], x_candidate[2,1]),
    label = false,
    mc =:red,
    z_order=:front)

    # make GIF
    if(ANIMATE)
        anim = @animate for i = 1:T
            animate_frame(i)
        end

        if(RUN_CM)
            gif(anim, "Saved_plots/cm.gif",fps=10)
        elseif(RUN_SSA)
            gif(anim, "Saved_plots/ssa.gif",fps=10)
        end
    end

    # make plot
    for i = 1:T
        animate_frame(i)
    end
    if(RUN_CM)
        savefig("Saved_plots/cm_plot.png")
    elseif(RUN_SSA)
        savefig("Saved_plots/ssa_plot.png")
    end

    plot(ssa_data, label = "State Selection Algorithm", shape =:utriangle, ms = 5)
    plot!(cm_data, label = "Conditional Mean", shape =:circle, ms = 4)
    savefig("Saved_plots/violation_rates.png")

end



