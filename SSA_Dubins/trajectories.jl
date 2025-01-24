using CUDA, LinearAlgebra, Distributions, Ipopt

function xprime!(ssa_params, Ξ::Array{Float64}, u::Array{Float64}, __pursuer::Dubins_Car, __evader::Dubins_Car)
    L = ssa_params.L
    ssa_particle_model = Model(Ipopt.Optimizer)
    set_silent(ssa_particle_model)

    for i in 1:L
        __,__,__,ω_opt = solve_particle_OCP(ssa_particle_model, ssa_params, __pursuer, __evader, Ξ[:,i])
        u[i,:] = ω_opt[1:N]
    end
end

function xk2prime!(ssa_params, __evader, u, w, xk2prime, i)
    M = ssa_params.M
    N = ssa_params.N

    Threads.@threads for j in 1:M
        for t = 1:N-1
            xk2prime[:,j,t+1] = dynamics.f(xk2prime[:,j,t],u[i,t],w[:,j,t])
        end
    end
end

####
# function: gpu_generate_Xi
# input: 
#   - L = number of particles
#   - n = number of dimensions of the state
#   - μ = mean state  
#   - var = covariance
# output: Ξ₀ = Array of randomly sampled states, size: [2 x L]
###
function gpu_generate_Xi(L :: Int64, n :: Int64, μ::Vector{Float64}, var)
    # Gaussian Density with mean vector μ_x0 and covariance matrix Σ_x0
    μ_x0 = CuArray(μ)
    Σ_x0 = (var*I)

    # randomly sample initial states Ξ following Gaussian density
    Ξ₀ = CuArray{Float64}(undef,n,L)
    Ξ₀ = μ_x0.+sqrt(Σ_x0)*CUDA.randn(n,L)
    return (Ξ₀)
end