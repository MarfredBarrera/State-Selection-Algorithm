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
M = 1500
N = 5
L = 1500

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
# fill state array with intial particle density
state[:,:,1] = Ξ

# initalize input array
u = CUDA.fill(0.0f0, L,N)

# generate random noise sequence Wprime for time horizon N for 
# state density with num particles L
w = (gpu_sample_gaussian_distribution(0, ω, (2,L,N)))
w2 = (gpu_sample_gaussian_distribution(0, ω, (2,L,N)))

# ### First, lets generate the x' trajectories for time horizon N for each particle in state density Xi ###
launch_xprime_kernel!(state, N, w, u)


cost = CUDA.fill(0.0f0, L)
state_2prime = CUDA.fill(0.0f0, (2,M,N))



function xk2prime(SSA_params, Ξ, state, u, w2)

    L = SSA_params.L
    M = SSA_params.M 
    N = SSA_params.N

    for i = 1:L
        # for each particle in the state density, randomly sample M particles
        local mc_sample_index = (rand(1:L, M))
        state[:,:,1] = Ξ[:,mc_sample_index]

        # calculate M sampled trajectories
        kernel = @cuda launch=false monte_carlo_sampling_kernel!(N, M, Ξ, state, u, w2,i)
        config = launch_configuration(kernel.fun)
        threads = min(length(state), config.threads)
        blocks = cld(length(state), threads)

        CUDA.@sync begin
            kernel(N, M, Ξ, state, u, w2, i; threads, blocks)
        end
    end
end

@benchmark xk2prime(SSA_params, Ξ, state_2prime, u, w2)
