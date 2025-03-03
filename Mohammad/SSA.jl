mutable struct SSA
    PF::ParticleFilter
    K₀::Function
    N::Int # prediction horizon length
    M::Int # number of monte carlo samples
    running_cost::Function
    check_constraint_violation::Function
    α::Float64 # constraint violation threshold
end

function SSA_sample_averages(SSA::SSA)
    n = size(SSA.PF.particles, 1)
    L = size(SSA.PF.particles, 2)
    X_prime = Array{Float64}(undef, (n, SSA.N, L))
    α_t_achieved = Array{Float64}(undef, (L, SSA.N))
    cost_t_achieved = Array{Float64}(undef, (L, SSA.N))
    for i = 1:L
        X_prime[:,1,i] = SSA.PF.particles[:,i]
        x_dprime_per_sample = SSA.PF.particles[:,rand(1:L, SSA.M)]
         for t = 1:SSA.N-1
            u = SSA.K₀(X_prime[:,t,i])
            X_prime[:,t+1,i] = SSA.PF.f(X_prime[:,t,i], u)
            cost_t = 0.0
            α_t = 0.0
            for j = 1:SSA.M
            x_dprime_per_sample[:,j] = SSA.PF.f(x_dprime_per_sample[:,j], u)
            cost_t += SSA.running_cost(x_dprime_per_sample[:,j], u)
            α_t += SSA.check_constraint_violation(x_dprime_per_sample[:,j])
            end
            α_t_achieved[i,t] = α_t / SSA.M
            cost_t_achieved[i,t] = cost_t / SSA.M
        end
    end
    return X_prime[:,1,:], α_t_achieved, cost_t_achieved
end

function SSA_select(SSA::SSA, x_prime_0, α_t_achieved, cost_t_achieved)
    n = size(SSA.PF.particles, 1)
    L = size(SSA.PF.particles, 2)
    feasible_indices = falses(L)
    cost_achieved = zeros(L)
    for i = 1:L
        # Check feasibility
        if all(α_t_achieved[i,:] .<= SSA.α)
            feasible_indices[i] = true
        end
        # Check predicted cost
        #print(cost_t_achieved[i,:])
        cost_achieved[i] = sum(cost_t_achieved[i,:])
    end
    feasible_costs = cost_achieved[feasible_indices]
    feasible_indices_set = findall(feasible_indices)
    if isempty(feasible_costs)
        println("No feasible state found!")
        α_achieved_sum = sum(α_t_achieved, dims=2)
        min_α, min_index = findmin(α_achieved_sum)
        return x_prime_0[:,min_index[0]]
    else
        min_cost, min_index = findmin(cost_achieved[feasible_indices])
        return x_prime_0[:, feasible_indices_set[min_index]]
    end
end