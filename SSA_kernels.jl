
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


# function: master_kernel!
# objective: from M number of x'' trajectories, calculate cost and violation rates simultaneously
function master_kernel!(SSA_limits, T, M, state, u, cost, state_violation_rate, control_violation_rate, i)
    cost_kernel!(T,M,state,u,cost,i)
    constraint_violation_kernel!(SSA_limits, T,M,state,u,state_violation_rate, control_violation_rate, i)
    return
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

    return
end

# function: constraint_violation_kernel!
# objective: calculate constraint violation rates
function constraint_violation_kernel!(SSA_limits, T,M,state,u,state_violation_rate, control_violation_rate, i)
    # unpack SSA_limits struct
    Ulim = SSA_limits.Ulim
    x1_upperlim = SSA_limits.x1_upperlim
    x1_lowerlim = SSA_limits.x1_lowerlim
    y1_upperlim = SSA_limits.y1_upperlim
    y1_lowerlim = SSA_limits.y1_lowerlim
    x2_upperlim = SSA_limits.x2_upperlim
    x2_lowerlim = SSA_limits.x2_upperlim
    y2_upperlim = SSA_limits.y2_lowerlim
    y2_lowerlim = SSA_limits.y2_upperlim

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # compare each trajectory with input and state constraints
    for j = index:stride:M
        for t = 1:T
            if(u[i,t] > Ulim)
                control_violation_rate[i] += 1
            end
            if((state[1,j,t] > x1_lowerlim && state[1,j,t] < x1_upperlim) ||
                (state[2,j,t] > y1_lowerlim && state[2,j,t] < y1_upperlim) ||
                (state[1,j,t] > x2_lowerlim && state[1,j,t] < x2_upperlim) ||
                (state[2,j,t] > y2_lowerlim && state[2,j,t] < y2_upperlim))
                state_violation_rate[i] += 1
            end
        end
    end
    return
end