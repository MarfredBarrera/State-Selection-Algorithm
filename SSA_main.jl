include("SSA_utils.jl")
include("SSA_particle_filter.jl")

struct Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int32
end
struct Gaussians
    # state density with mean and variance
    μ :: Float64
    Σ :: Float64

    # process noise ωₖ and scalar measurement noise vₖ
    ω :: Float64
    v :: Float64
end
struct Limits
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
M = 1000
N = 5
L = 3000
n = 2

# state density mean and variance
μ = 7.5 
Σ = 0.5

# process noise ωₖ and scalar measurement noise vₖ
ω = 0.5
v = 0.5

Ulim = 3
α = 0.05
ϵ = 0.15
δ = 0.001

# state constraints
x1_upperlim = 5
x1_lowerlim = 3
y1_upperlim = 2
y1_lowerlim = -4

x2_upperlim = 5
x2_lowerlim = -2
y2_upperlim = -4
y2_lowerlim = - 7
 

# store parameters in struct
SSA_params = Params(M, N, L, n)
SSA_gauss  = Gaussians(μ, Σ, ω, v)
SSA_limits = Limits(Ulim, α, ϵ, δ, 
    x1_upperlim, x1_lowerlim, 
    y1_upperlim, y1_lowerlim,
    x2_upperlim, x2_lowerlim, 
    y2_upperlim, y2_lowerlim)

# generate state density Xi according to Gaussian parameters
(Ξ_gaussian, W_gaussian, V_gaussian) = init_gaussians()
Ξ = gpu_generate_Xi(SSA_params.L, SSA_params.n)

@benchmark CUDA.@sync candidate_state = state_selection_algorithm(Ξ,SSA_params,SSA_limits)


