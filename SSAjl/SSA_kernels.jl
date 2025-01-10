
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
        
    for i = index:stride:size(u,1)
        @inbounds u[i,1] = -0.05*state[1,i,1]*state[2,i,1]
        for t ∈ 2:T
            @inbounds begin
                u[i,t] = -0.05*state[1,i,t]*state[2,i,t]
                state[1,i,t+1] = 0.9*state[1,i,t] + 0.2*state[2,i,t] + w[1,i,t]
                state[2,i,t+1] = -0.15*state[1,i,t] + 0.9*state[2,i,t] + 0.05*state[1,i,t]*state[2,i,t] + u[i,t] + w[2,i,t]
            end
        end
    end
    return
end


# function: master_kernel!
# inputs: 
#       SSA_limits - struct of state/input constraints
#       T - time horizon
#       state - [2 x M x N]; array of Monte Carlo sampled trajectories
#       u - [L x N]; array of input sequences for entire particle density
#       sampled_cost - [M x N]; array with cost at each time step
#       state_violation_count - [M x N]; array with 1 at each element that violates state constraint
#       i - for loop iterator
# objective: from M number of x'' trajectories, calculate cost and violation rates simultaneously
function master_kernel!(SSA_limits, T, M, state, u, sampled_cost, state_violation_count, i)
    cost_kernel!(T,M,state,u,sampled_cost,i)
    constraint_violation_kernel!(SSA_limits, T,M,state,u,state_violation_count, i)
    return
end

# function: cost_kernel! - calculates the cost of M sampled trajectories for one particle
# inputs:
#       T - time horizon
#       M - number of samples
#       u - [L x N] array of input sequences
#       sampled_cost - [M x N] array with cost at each time step
# objective: fill out sampled_cost with cost at each time step, then sum it outside of cost_kernel!
function cost_kernel!(T,M,state,u,sampled_cost,i)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for j in index:stride:M
        for t in 1:T
            @inbounds sampled_cost[j,t] = state[1,j,t]^2 + state[2,j,t]^2
        end
    end

    return
end

# function: constraint_violation_kernel!
# objective: calculate constraint violation rates
function constraint_violation_kernel!(SSA_limits,T,M,state,u, state_violation_count, i)
    # unpack SSA_limits struct
    Ulim = SSA_limits.Ulim
    x1_upperlim = SSA_limits.x1_upperlim
    x1_lowerlim = SSA_limits.x1_lowerlim
    y1_upperlim = SSA_limits.y1_upperlim
    y1_lowerlim = SSA_limits.y1_lowerlim
    x2_upperlim = SSA_limits.x2_upperlim
    x2_lowerlim = SSA_limits.x2_lowerlim
    y2_upperlim = SSA_limits.y2_upperlim
    y2_lowerlim = SSA_limits.y2_lowerlim

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # compare each trajectory with state constraints
    for j = index:stride:M
        for t = 1:T

            in_region1 = (x1_lowerlim < state[1,j,t] < x1_upperlim) && (y1_lowerlim < state[2,j,t] < y1_upperlim)
            in_region2 = (x2_lowerlim < state[1,j,t] < x2_upperlim) && (y2_lowerlim < state[2,j,t] < y2_upperlim)

            if(in_region1||in_region2)
                state_violation_count[j,t] = 1
            end
        end
    end


    return 
end