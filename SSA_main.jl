include("SSA_utils.jl")

struct Params
    M :: Int64
    N :: Int64
    L :: Int64
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
end

# intialize parameters
M = 300
N = 5
L = 2^15

# state density mean and variance
μ = 7.5 
Σ = 0.5

# process noise ωₖ and scalar measurement noise vₖ
ω = 0.5
v = 0.5

Ulim = 3
α = 0.1
ϵ = 0.3
δ = 0.001

# store parameters in struct
SSA_params = Params(M, N, L)
SSA_gauss  = Gaussians(μ, Σ, ω, v)
SSA_limits = Limits(Ulim, α, ϵ, δ)

# generate state density Xi according to Gaussian parameters
(Ξ_gaussian, W_gaussian, V_gaussian) = init_gaussians()
Ξ = gpu_generate_Xi(SSA_params.L)


# intialize state array
state = CUDA.fill(1.0f0, (2,L,N))
u = CUDA.fill(0.0f0, L)

# generate random noise sequence Wprime for time horizon N for 
# state density with num particles L
w = (gpu_sample_gaussian_distribution(0, ω, (2,L,N)))


# ### First, lets generate the x' trajectories for time horizon N for each particle in state density Xi ###

@benchmark launch_xprime_kernel!($state, $N, $w)
