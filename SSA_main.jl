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
M = 1000
N = 5
L = 1000

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

t1 = time()
for i = 1:L
    xk2prime!(SSA_params, Ξ, state_2prime, u, w2, i)
    # TODO: check feasability/constraint violations
    launch_cost_kernel!(N, M, state_2prime, u, cost, i)
end
elapsed_time = time()-t1
println(elapsed_time)

findmin(cost)