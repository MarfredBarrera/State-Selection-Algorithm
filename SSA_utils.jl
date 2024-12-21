# import Packages
using CUDA
using Test
using BenchmarkTools
using Random
using LinearAlgebra
using Distributions

include("SSA_kernels.jl")

####
# function: gpu_generate_Xi
# input: L = number of particles
# output: Ξ₀ = Array of randomly sampled states, size: [2 x L]
###
function gpu_generate_Xi(L :: Int64, n :: Int64, μ)
    # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
    μ_x0 = CuArray(μ)
    Σ_x0 = (0.5*I)

    # randomly sample initial states Ξ following Gaussian density
    Ξ₀ = CuArray{Float64}(undef,n,L)
    Ξ₀ = μ_x0.+sqrt(Σ_x0)*CUDA.randn(n,L)
    return (Ξ₀)
end

function gpu_sample_gaussian_distribution(mean, var, dims)
    w = CuArray{Float64}(undef, dims[1],dims[2],dims[3])
    w = mean.+sqrt(var)*CUDA.randn(dims[1],dims[2],dims[3])
    return w
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

# function: launch_constraint_kernel!
# objective: launch kernel for violation rate calculation
function launch_constraint_kernel!(SSA_limits,T,M,state,u, state_violation_count, i)
    kernel = @cuda launch=false constraint_violation_kernel!(SSA_limits,T,M,state,u, state_violation_count, i)
    config = launch_configuration(kernel.fun)
    threads = min(M, config.threads)
    blocks = cld(M, threads)

    CUDA.@sync begin
        kernel(SSA_limits,T,M,state,u, state_violation_count, i; threads, blocks)
    end
end

# function: xk2prime
# input:
# - N: time horizon
# - M: Monte Carlo sample number 
# - u: array of control sequences [1 X L]
# - w: array of random noise [2 x M x N]
# - xk2prime: batch of Monte Carlo sampled trajectories
function xk2prime!(N, M, u, w, xk2prime, i)
    for j = 1:M
        for t = 1:N-1
            xk2prime[:,j,t+1] = dynamics.f(xk2prime[:,j,t],u[i,t],w[:,j,t])
        end
    end
end

# function: state_selection_algorithm
# inputs:
# - Ξ: particle density
# - SSA_params: struct of parameters
# - SSA_limits: struct of state and input constraints
function state_selection_algorithm(Ξ,SSA_params,SSA_limits)
    n = SSA_params.n
    L = SSA_params.L
    N = SSA_params.N
    M = SSA_params.M

    # intialize state array
    state = CUDA.fill(1.0f0, (n,L,N))
    # fill state array with intial particle density
    state[:,:,1] = Ξ

    # initalize input array
    u = CUDA.fill(0.0f0, L,N)

    # generate random noise sequence Wprime for time horizon N for 
    # state density with num particles L
    w = (gpu_sample_gaussian_distribution(0, ω, (n,L,N)))
    w2 = Array(gpu_sample_gaussian_distribution(0, ω, (n,L,N)))

    # ### First, lets generate the x' trajectories for time horizon N for each particle in state density Xi ###
    CUDA.@sync launch_xprime_kernel!(state, N, w, u)

    # declare vectors for x'' trajectories, cost, and constraint violation rate calculations
    cost = fill(0.0f0, L)
    sampled_costs = CUDA.fill(0.0f0, M,N)
    state_violation_count = CUDA.fill(0.0f0, M,N)
    sampled_state_violations = CUDA.fill(0.0f0,L,N)
    sampled_control_violations = CUDA.fill(0.0f0,L,N)
    total_state_violations = fill(false,L)
    total_control_violations = fill(false,L)
    state_2prime = CUDA.fill(0.0f0, (n,M,N))


    Ξ = CuArray(Ξ)
    # iterate through each particle in Ξ and run M monte carlo simulations for each particle 
    for i = 1:L
        CUDA.@sync begin
            # calculate x'' trajectories

            mc_sample_index = (rand(1:L, M))
            state_2prime[:,:,1] = Ξ[:,mc_sample_index]
            xk2prime!(N, M, Array(u), w2, Array(state_2prime), i)
            # launch_xk2prime_kernel!(SSA_params, state_2prime, u, w2, i)

            # calculate cost and state/control violation rates
            launch_constraint_kernel!(SSA_limits, N, M , state_2prime, u, state_violation_count, i)

            # sum the sampled cost to calculate the cost of each L particles
            cost[i] = M*sum(state[:,i,:].^2) + sum(state_2prime.^2)

            # sum the violation counts to make an [L x N] array, which contains the total violations of each trajectory
            sampled_state_violations[i,:] = sum(state_violation_count, dims=1)

            # indicate which particles satisfy state constraints
            total_state_violations[i] = all(sampled_state_violations[i,:]/M .< α)
            # total_control_violations[i] = all(sampled_control_violations[i,:]/M .< α)
        end
    end

    # mask for feasible states
    feasibility_mask = total_state_violations

    u = Array(u)

    
    if(sum(feasibility_mask)==0) # if there are no feasible states, the feasible set is empty and SSA cannot proceed
        println("Feasible set is empty!")
        cost_val,candidate_index = findmin(cost)
        return Array(Ξ)[:,candidate_index], u[candidate_index,1], candidate_index, feasibility_mask
    else # otherwise, find feasible state with minimum cost
        cost_val, candidate_index = findmin(cost[feasibility_mask])
        # println(cost_val,candidate_index, Array(Ξ)[:,candidate_index])
        return Array(Ξ)[:,candidate_index], u[candidate_index,1], candidate_index, feasibility_mask
    end

    return
end


## function: run_simulation(T)
# input:
#       T - run simulation for T time steps
# objective: run the bootstrap particle filter in conjunction with the SSA/CM for T time steps
function run_simulation(T)

    
    sim_data = fill(0.0f0, (n,L,T))
    violation_rate = fill(0.0f0,T)
    x_candidate = fill(0.0f0, (n,T))

    # initialize and store true state
    x_true = Array{Float64}(undef, n, T+1)
    x_true[:,1] = μ.+ sqrt(Σ)*randn(2)

    # generate state density Xi according to Gaussian parameters
    Ξ = gpu_generate_Xi(SSA_params.L, SSA_params.n,μ)

    # set intial particle density of the bootstrap particle filter
    pf.particles = Array(Ξ)

    # start simultion loop with T time steps
    for t = 1:T
        sim_data[:,:,t] = pf.particles

        if(RUN_SSA) # run the state selection algorithm for the particle density
            CUDA.@sync candidate_state, u_star, candidate_index, feasibility_mask = state_selection_algorithm(pf.particles,SSA_params,SSA_limits)
            x_candidate[:,t] = pf.particles[:,candidate_index]
            if(isinf(u_star)||isnan(u_star))
                println("Feasible Set is Empty!!")
                break
            end
        elseif(RUN_CM) # choose the conditional mean as the state estimates
            x_candidate[:,t] = mean(pf.particles, dims = 2)
            candidate_state = x_candidate[:,t]
        else
            error("Please choose a state selection type to simulate")
        end

        # check how many particles violate state constraints
        violation_rate[t] = sum(check_constraints(pf.particles))/L
        println("Violation Rate: ", violation_rate[t])

        # controller based on selected_state
        u_star = dynamics.u(candidate_state)
    
        ### BOOTSTRAP PARTICLE FILTER UPDATE ###

        # generate random noise
        w = Array(gpu_sample_gaussian_distribution(0, ω, (n,L,1)))
        w_true = sqrt(ω)*randn(2)
    
        # propagate particle density
        pf.particles = pf.TimeUpdate(pf.particles, dynamics, u_star, w)
       
        # propagate true state
        x_true[:,t+1] = dynamics.f(x_true[:,t], u_star, w_true)
    
        # take measurement of true state
        y = dynamics.h(x_true[:,t+1], sqrt(v)*randn())
    
        # calculate likelihoods of states based on measurement
        pf.MeasurementUpdate(pf,dynamics,y)
    
        # resample with these new likelihoods
        pf.Resampler(pf)
    end

    return x_candidate, sim_data, violation_rate
end


# function: check_constraints
# input: x - 2D row vector of state particles
# output: 1 x length(x) vector of 1s and 0s, 1 being a state violation
function check_constraints(x)
    constraint_count = fill(0.0f0,size(x,2))
    for i = eachindex(constraint_count)
        in_region1 = (x1_lowerlim < x[1,i] < x1_upperlim) && (y1_lowerlim < x[2,i] < y1_upperlim)
        in_region2 = (x2_lowerlim < x[1,i] < x2_upperlim) && (y2_lowerlim < x[2,i] < y2_upperlim)

        if(in_region1||in_region2)
            constraint_count[i] = 1
        end
    end
    return constraint_count
end
