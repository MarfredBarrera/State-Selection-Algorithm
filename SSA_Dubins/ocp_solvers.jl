using JuMP

# function: solve_evader_OCP
# objective: given the state of an evader and state of a pursuer, find the optimal control for the evader to avoid the pursuer 
# inputs: 
#   - model: OCP solver model
#   - sim_params: struct of simulation parameters
#   - __pursuer: dubins car object for pursuer
#   - __evader: dubins car object for evader
function solve_evader_OCP(model, sim_params, __pursuer, __evader)


    N = sim_params.N
    Δt = sim_params.Δt


    x_pursuer = __pursuer.x_true[1]
    y_pursuer = __pursuer.x_true[2]

    x0 = __evader.x_measured[1]
    y0 = __evader.x_measured[2]
    θ0 = __evader.x_measured[3]

    v = __evader.v
    u_max = __evader.u_max

    # Decision variables
    @variable(model, x[1:N+1])      # x position
    @variable(model, y[1:N+1])      # y position
    @variable(model, θ[1:N+1])      # orientation
    @variable(model, ω[1:N])        # control input (turning rate)

    # Objective function (quadratic cost on position and control) --> maximize distance to pursuer
    @objective(model, Min, 
        -sum((x[k] - x_pursuer)^2 + (y[k] - y_pursuer)^2 for k in 1:N)) + sum(ω[k]^2 for k in 1:N)

    # Initial conditions
    @constraint(model, x[1] == x0)
    @constraint(model, y[1] == y0)
    @constraint(model, θ[1] == θ0)

    # Dubins car dynamics constraints
    for k in 1:N
        @constraint(model, x[k+1] == x[k] + v * cos(θ[k]) * Δt)
        @constraint(model, y[k+1] == y[k] + v * sin(θ[k]) * Δt)
        @constraint(model, θ[k+1] == θ[k] + ω[k] * Δt)

        # bounded control input
        @constraint(model, -u_max <= ω[k] <= u_max)

    end

    # Solve the optimization problem
    optimize!(model)

    # Extract results
    x_opt = value.(x)
    y_opt = value.(y)
    θ_opt = value.(θ)
    ω_opt = value.(ω)

    return x_opt,y_opt,θ_opt,ω_opt
end

# function: solve_pursuer_OCP
# objective: given the state of an evader and state of a pursuer, find the optimal control for the pursuer to catch the evader 
# inputs: 
#   - model: OCP solver model
#   - sim_params: struct of simulation parameters
#   - __pursuer: dubins car object for pursuer
#   - __evader: dubins car object for evader
function solve_pursuer_OCP(model, sim_params, __pursuer, __evader)
    N = sim_params.N
    Δt = sim_params.Δt

    x_evader = __evader.x_true[1]
    y_evader = __evader.x_true[2]
    θ_evader = __evader.x_true[3]

    x0 = __pursuer.x_true[1]
    y0 = __pursuer.x_true[2]
    θ0 = __pursuer.x_true[3]

    v = __pursuer.v
    u_max = __pursuer.u_max

    # Decision variables
    @variable(model, x[1:N+1])      # x position
    @variable(model, y[1:N+1])      # y position
    @variable(model, θ[1:N+1])      # orientation
    @variable(model, ω[1:N])        # control input (turning rate)

    # Objective function (quadratic cost on position and control)
    @objective(model, Min, 
        sum((x[k] - x_evader)^2 + (y[k] - y_evader)^2 for k in 1:N))

    # Initial conditions
    @constraint(model, x[1] == x0)
    @constraint(model, y[1] == y0)
    @constraint(model, θ[1] == θ0)

    # Dubins car dynamics constraints
    for k in 1:N
        @constraint(model, x[k+1] == x[k] + v * cos(θ[k]) * Δt)
        @constraint(model, y[k+1] == y[k] + v * sin(θ[k]) * Δt)
        @constraint(model, θ[k+1] == θ[k] + ω[k] * Δt)

        # bounded control input
        @constraint(model, -u_max <= ω[k] <= u_max)
    end
    
    # Solve the optimization problem
    optimize!(model)

    # Extract results
    x_opt = value.(x)
    y_opt = value.(y)
    θ_opt = value.(θ)
    ω_opt = value.(ω)

    return x_opt,y_opt,θ_opt,ω_opt
end

function solve_particle_OCP(model, ssa_params, __pursuer::Dubins_Car, __evader::Dubins_Car, particle_state::Vector{Float64})
    N = ssa_params.N
    Δt = ssa_params.Δt


    x_pursuer = __pursuer.x_true[1]
    y_pursuer = __pursuer.x_true[2]

    # solve the optimal control problem assuming we start at the given state particle
    x0 = particle_state[1]
    y0 = particle_state[2]
    θ0 = particle_state[3]

    v = __evader.v
    u_max = __evader.u_max

    # Decision variables
    x = @variable(model, [1:N+1])      # x position
    y = @variable(model, [1:N+1])      # y position
    θ = @variable(model, [1:N+1])      # orientation
    ω = @variable(model, [1:N])        # control input (turning rate)

    # Objective function (quadratic cost on position and control) --> maximize distance to pursuer
    @objective(model, Min, 
        -sum((x[k] - x_pursuer)^2 + (y[k] - y_pursuer)^2 for k in 1:N) + sum(ω[k]^2 for k in 1:N))

    # Initial conditions
    @constraint(model, x[1] == x0)
    @constraint(model, y[1] == y0)
    @constraint(model, θ[1] == θ0)

    # Dubins car dynamics constraints
    for k in 1:N
        @constraint(model, x[k+1] == x[k] + v * cos(θ[k]) * Δt + randn()*Σ)
        @constraint(model, y[k+1] == y[k] + v * sin(θ[k]) * Δt + randn()*Σ)
        @constraint(model, θ[k+1] == θ[k] + ω[k] * Δt)

        # bounded control input
        @constraint(model, -u_max <= ω[k] <= u_max)

    end

    # Solve the optimization problem
    optimize!(model)

    # Extract results
    x_opt = value.(x)
    y_opt = value.(y)
    θ_opt = value.(θ)
    ω_opt = value.(ω)

    return x_opt,y_opt,θ_opt,ω_opt
end