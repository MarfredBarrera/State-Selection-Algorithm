using JuMP
using HiGHS  # You can replace HiGHS with OSQP or another solver of your choice
using LinearAlgebra
using BenchmarkTools

# Define the number of particles
L = 1000  # Number of particles

# Problem dimensions
n = 5    # Dimension of the control variable (u)

# Generate shared problem data
Q = Matrix{Float64}(I, n, n)   # Quadratic cost matrix (identity matrix)
c = randn(n,L)                   # Linear cost vector (randomized)
A = vcat(ones(1, n), -ones(1, n))  # Inequality constraint matrix
b = [1.0; -1.0]                # Inequality constraint bounds

# Define a function to solve the QP for a single particle
function solve_qp_for_particle(i,c_tot)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Define variables
    @variable(model, u[1:n])

    c = c_tot[:,i]

    # Define the objective function
    @objective(model, Min, 0.5 * u' * Q * u + c' * u)

    # Define constraints
    @constraint(model, A * u .<= b)

    # Solve the optimization problem
    optimize!(model);

    # Check solver status and return results
    if termination_status(model) == MOI.OPTIMAL
        return value.(u)
    else
        println("Particle $i: Solution not optimal.")
        return nothing
    end
end



function solve_particle_density(L)
    # Solve the QP for all particles in parallel
    results = Vector{Union{Vector{Float64}, Nothing}}(undef, L)  # Store results for each particle
    Threads.@threads for i in 1:L
        results[i] = solve_qp_for_particle(i,c);
    end
    return results
end

@btime results = solve_particle_density(L)
