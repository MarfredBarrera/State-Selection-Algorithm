

function xprime!(u, SSA_params, state, Σ::Dynamics, kd::Function)
    Threads.@threads for i = 1:SSA_params.L
         for t = 1:SSA_params.N-1
              u[:,t,i] = kd(state[:,t,i])
              state[:,t+1,i] = dynamics(Σ, state[:,t,i] ,u[:,t,i])
         end
    end
end

function x2prime!(u, SSA_params, state, Σ::Dynamics, kd::Function)
    for i = 1:SSA_params.M
         for t = 1:SSA_params.N-1
              state[:,t+1,i] = dynamics(Σ, state[:,t,i], u[:,t])
         end
    end
end

function state_selection_algorithm(SSA_params, Ξ, kd::Function, Σ::Dynamics, J::Function)
    n = SSA_params.n
    N = SSA_params.N
    M = SSA_params.M
    L = SSA_params.L


    state = Array{Float64}(undef, (n,N,L))
    u_total = Array{Float64}(undef, (m,N,L))
    state_2prime = Array{Float64}(undef,(n,N,M))
    cost = Array{Float64}(undef, L)
    sampled_state_violation_count = zeros(M)
    total_state_violations = Array{Float64}(undef, L)


    state[:,1,:] = Ξ
    

    xprime!(u_total, SSA_params, state, Σ, kd)

    Threads.@threads for i = 1:L
        mc_sample_index = (rand(1:L, M))
        state_2prime[:,1,:] .= Ξ[:,mc_sample_index]

        u_j = u_total[:,:,i]
        x2prime!(u_j, SSA_params, state_2prime, Σ, kd)

        for j = 1:M
            for t = 1:N
                cost[i] += J(state_2prime[:,t,j], u_j[:,t]) + J(state[:,t,i], u_j[:,t])
                # if(check_constraints(state_2prime[:,t,j]) == 1)
                #     sampled_state_violation_count[j] = 1.0
                # end
            end
        end
        # total_state_violations[i] = sum(sampled_state_violation_count)/M
    end

    # if(sum(feasibility_mask) <= 0)
    #     println("Feasible set is empty!")
    #     cost_val, candidate_index = findmin(cost)
    #     return Ξ[:,candidate_index]
    # else
    #     cost_val, candidate_index = findmin(cost[feasibility_mask])
    #     return Ξ[:,candidate_index]
    # end

    cost_val, candidate_index = findmin(cost)
    return Ξ[:,candidate_index]
end

check_constraints(x) = h(x) > 0 ? 0 : 1