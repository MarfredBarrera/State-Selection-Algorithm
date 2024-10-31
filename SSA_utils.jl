# import Packages
using CUDA
using Test
using BenchmarkTools
using Random
using LinearAlgebra
using Distributions

# TODO: have all parameters (covariances, particles, etc...) set from JSON file

# function: init_gaussians() = initialize particle cloud, process noise, 
# and measurement noise gaussian distributions for random sampling during SSA
# 
# input: none
# output: tuple of Normal objects from Distributons Pkg 
function init_gaussians()
    μ = 7.5
    Σ = 0.5

    # process noise ωₖ and scalar measurement noise vₖ
    ω = 0.5
    v = 0.5

    Ξ_gaussian = Normal(μ, sqrt(Σ))
    W_gaussian = Normal(0,sqrt(0.5))
    V_gaussian = Normal(0,sqrt(0.5))

    return (Ξ_gaussian, W_gaussian, V_gaussian)
end


####
# function: gpu_generate_Xi
# input: L = number of particles
# output: Ξ₀ = Array of randomly sampled states, size: [2 x L]
###
function gpu_generate_Xi(L :: Int64)
    # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
    μ_x0 = CuArray([7.5,-7.5])
    Σ_x0 = (0.5*I)

    # randomly sample initial states Ξ following Gaussian density
    Ξ₀ = CuArray{Float64}(undef,2,L)
    Ξ₀ = μ_x0.+sqrt(Σ_x0)*CUDA.randn(2,L)
    return (Ξ₀)
end

function gpu_sample_gaussian_distribution(mean, var, dims)
    w = CuArray{Float64}(undef, dims[1],dims[2],dims[3])
    w = mean.+sqrt(var)*CUDA.randn(dims[1],dims[2],dims[3])
    return w
end

# function: cpu_generate_Xi
# version of gpu_generate_Xi to run on cpu for benchmark comparisons
function cpu_generate_Xi(L :: Int64)
    # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
    μ_x0 = ([7.5,-7.5])
    Σ_x0 = (0.5*I)

    # randomly sample initial states Ξ following Gaussian density
    Ξ₀ = Array{Float64}(undef,2,L)
    Ξ₀ = μ_x0.+sqrt(Σ_x0)*randn(2,L)
    return Array(Ξ₀)
end


# function: xprime_kernel_function!
# inputs: 
#   state - [2 x L x N] state array
#   T - propagate up to time T
#   w - randomly generated noise
#   u - input [L x N]
#
# objective: propagate state estimates according to example dynamics, control, and noise using GPU
function xprime_kernel_function!(state, T, w, u)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
        for t ∈ 1:T-1
            for i = index:stride:length(state)
                @inbounds u[i,t] = -0.05*state[1,i,t]*state[2,i,t]
                @inbounds state[1,i,t+1] = 0.9*state[1,i,t] + 0.2*state[2,i,t] + w[1,i,t]
                @inbounds state[2,i,t+1] = -0.15*state[1,i,t] + 0.9*state[2,i,t] + 0.05*state[1,i,t]*state[2,i,t] + u[i,t] + w[2,i,t]
            end
        end 
 
        for i = index:stride:length(state)
            @inbounds u[i,T] = -0.05*state[1,i,T]*state[2,i,T]
        end
    return
end


# function: launch_xprime_kernel
# inputs: 
#   state - [2 x L x N] state array
#   T - propagate up to time T
#   w - randomly generated noise
# 
# objective: configure and launch the xprime kernel function
function launch_xprime_kernel!(state, T, w, u)
    kernel = @cuda launch=false xprime_kernel_function!(state, T, w, u)
    config = launch_configuration(kernel.fun)
    threads = min(length(state), config.threads)
    blocks = cld(length(state), threads)

    CUDA.@sync begin
        kernel(state, T, w, u; threads, blocks)
    end
end


## function: monte_carlo_sampling_kernel - kernel function that calculates M samples for one particle
# inputs: T - time steps, M - sample number, state - initial state density, i - iterator through L
#  u - input, w2 - randomly generated noise
#
# output: updated state array 
function monte_carlo_sampling_kernel!(T, M, Ξ, state, u, w2, i)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for t ∈ 1:T-1
        for j = index:stride:length(state)
            @inbounds state[1,j,t+1] = 0.9*state[1,j,t] + 0.2*state[2,j,t] + w2[1,j,t]
            @inbounds state[2,j,t+1] = -0.15*state[1,j,t] + 0.9*state[2,j,t] + 0.05*state[1,j,t]*state[2,j,t] + u[i,t] + w2[2,j,t]
        end
    end
end  


## function: xk2prime! - compute M number x'' sequences for one particle
# inputs: T - time steps, M - sample number, state - initial state density, i - iterator through L
#  u - input, w2 - randomly generated noise
#
# output: updated state array 
function xk2prime!(SSA_params, Ξ, state, u, w2, i)

    L = SSA_params.L
    M = SSA_params.M 
    N = SSA_params.N

    # for each particle in the state density, randomly sample M particles
    local mc_sample_index = (rand(1:L, M))
    state[:,:,1] = Ξ[:,mc_sample_index]

    # calculate M sampled trajectories
    kernel = @cuda launch=false monte_carlo_sampling_kernel!(N, M, Ξ, state, u, w2, i)
    config = launch_configuration(kernel.fun)
    threads = min(length(state), config.threads)
    blocks = cld(length(state), threads)

    CUDA.@sync begin
        kernel(N, M, Ξ, state, u, w2, i; threads, blocks)
    end
end

# cost kernel launcher
function launch_cost_kernel!(T, M, state, u, cost, i)
    kernel = @cuda launch=false cost_kernel!(T,M,state,u,cost,i)
    config = launch_configuration(kernel.fun)
    threads = min(M, config.threads)
    blocks = cld(M, threads)

    CUDA.@sync begin
        kernel(T,M,state,u,cost,i; threads, blocks)
    end
end

# function: cost_kernel! - calculates the cost of M sampled trajectories for one particle
function cost_kernel!(T,M,state,u,cost,i)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for j in index:stride:M
        for t in 1:T
            cost[i] += state[1,j,t]^2 + state[2,j,t]^2 + u[i,t]^2
        end
    end
end