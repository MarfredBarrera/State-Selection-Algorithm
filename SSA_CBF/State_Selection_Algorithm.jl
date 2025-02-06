

"""
function: xprime!
    objective: calculate open loop trajectories and control sequences for each particle
    inputs:
        u - [m x N x L] size array for storing control sequences for all particles
        SSA_params - struct of SSA parameters
        state - [n x N x L] size array for storing trajectories of all particles
        Σ - dynamics for a control affine system
        kd - control law for u = kd(x)
"""
function xprime!(u, SSA_params, state, Σ::Dynamics, kd::Function)
    Threads.@threads for i = 1:SSA_params.L
         for t = 1:SSA_params.N-1
              u[:,t,i] = kd(state[:,t,i])
              state[:,t+1,i] = dynamics(Σ, state[:,t,i] ,u[:,t,i])
         end
    end
end


"""
function: x2prime!
    objective: calculate trajectories for sampled particles given a control sequence from xprime
    inputs:
        u - [m x N] size array for the control sequence to be applied
        SSA_params - struct of SSA parameters
        state - [n x N x M] size array for storing trajectories of all sampled particles
        Σ - dynamics for a control affine system
        kd - control law for u = kd(x)
"""
function x2prime!(u, SSA_params, state, Σ::Dynamics, kd::Function)
    for j = 1:SSA_params.M
         for t = 1:SSA_params.N-1
                state[:,t+1,j] = dynamics(Σ, state[:,t,j], u[:,t])
         end
    end
end


"""
function: state_selection_algorithm
    objective: select a single particle from a particle density according to SSA
    inputs:
        SSA_params - struct of SSA parameters
        Ξ - [n x L] particle density
        kd - control law for u = kd(x)
        Σ - dynamics for a control affine system
        J - prescibed cost function
"""
function state_selection_algorithm(SSA_params, Ξ, kd::Function, Σ::Dynamics, J::Function)
    n = SSA_params.n
    N = SSA_params.N
    M = SSA_params.M
    L = SSA_params.L
    α = 0.20


    state = Array{Float64}(undef, (n,N,L))
    u_total = Array{Float64}(undef, (m,N,L))
    state_2prime = Array{Float64}(undef,(n,N,M))
    cost = Array{Float64}(undef, L)
    total_state_violations = zeros(L)
    sampled_state_violation_count = zeros(N,M)
    feasibility_mask = falses(L)
    

    # calculate xprime sequence
    state[:,1,:] = Ξ
    xprime!(u_total, SSA_params, state, Σ, kd)

    # for each particle, carry out monte carlo sampling
    Threads.@threads for i = 1:L

        if(check_constraints(state[:,1,i]) > 0)
            total_state_violations[i] = Inf
            continue
        end

        mc_sample_index = (rand(1:L, M))

        # randomly sample M particles
        state_2prime[:,1,:] = Ξ[:,mc_sample_index]

        # extract control sequence for ith particle
        u_j = u_total[:,:,i]

        # apply control sequence and calculate x2prime sequence
        x2prime!(u_j, SSA_params, state_2prime, Σ, kd)

        # calculate cost of given batch of monte carlo samples
        for j = 1:M
            for t = 1:N
                cost[i] += J(state_2prime[:,t,j], u_j[:,t]) + J(state[:,t,i], u_j[:,t])

                
                if(check_constraints(state_2prime[:,t,j]) == 1)
                    sampled_state_violation_count[t,j] = 1.0
                end
            end
        end

        # only include feasible states
        feasibility_mask[i] = all(sum(sampled_state_violation_count,dims=2)/M .< α) 
    end


    if(sum(feasibility_mask) <= 0)
        println("Feasible set is empty!")
        cost_val, candidate_index = findmin(cost)
        return Ξ[:,candidate_index]
    else
        cost_val, candidate_index = findmin(cost[feasibility_mask])
        return Ξ[:,candidate_index]
    end
end

check_constraints(x) = h(x) > 0 ? 0 : 1