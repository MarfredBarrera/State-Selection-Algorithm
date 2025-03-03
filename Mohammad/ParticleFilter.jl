
## Bootstrap Partilce Filter (additive Gaussian)
mutable struct ParticleFilter
    f::Function
    h::Function
    W::Matrix{Float64}
    V::Matrix{Float64}
    particles::Matrix{Float64} # n x L matrix of particles, where n is the state dimension and L is the number of particles
    likelihoods::Vector{Float64}
    function ParticleFilter(f, h, W, V, particles, likelihoods)
        if !isposdef(W)
            throw(ArgumentError("W must be a positive semi-definite matrix."))
        end
        if !isposdef(V)
            throw(ArgumentError("V must be a positive semi-definite matrix."))
        end
        new(f, h, W, V, particles, likelihoods)
    end
end

function time_update(PF::ParticleFilter, u::Vector{Float64})
    particles_plus = Matrix{Float64}(undef, size(PF.particles))
    Threads.@threads for i = axes(particles_plus,2)
        particles_plus[:,i] = PF.f(PF.particles[:,i], u)
    end
    return particles_plus
end

function measurement_update(PF::ParticleFilter, y::Vector{Float64})
    measurement_likelihoods = zeros(size(PF.particles,2))
    V_inv = inv(PF.V)
    Threads.@threads for i = axes(PF.particles,2)
        err = y - PF.h(PF.particles[:,i])
        measurement_likelihoods[i] = exp.(-1/2 * err' * V_inv *err)
    end
    # We assume resampling is done every time step, so no need to multiply with old likelihoods
    new_likelihoods = PF.likelihoods .* measurement_likelihoods # p(x) * p(y|x)
    PF.likelihoods = new_likelihoods ./ sum(new_likelihoods) # normalize
end

function resampler!(PF::ParticleFilter)
    particles_resampled = zeros(size(PF.particles))
    CDF = cumsum(PF.likelihoods)
    for i = axes(particles_resampled, 2)
        particles_resampled[:,i] = PF.particles[:,findfirst(CDF .>= rand(1))]
    end
    PF.particles = particles_resampled
end

function propagate_PF!(PF::ParticleFilter, u::Vector{Float64}, y::Vector{Float64})
    time_update(PF, u)
    measurement_update(PF, y)
    resampler!(PF)
end