## Bootstrap Partilce Filter (additive Gaussian)

Base.@kwdef mutable struct Particle_Filter
    model::Dynamics
    TimeUpdate::Function
    MeasurementUpdate::Function
    Resampler::Function
    likelihoods::Vector
    particles::Array
end

## function: TimeUpdate
# input: 
#       - x: set of particles in [n x L] matrix
#       - model: set of dynamics
#       - u: input
#       - w: randomly generated noise of same size as x, [n x L]
#       - Δt: time step
# output:
#       - x_plus: all particles propagated one time step
function TimeUpdate(pf, x, Σ, u)
    x_plus = Array{Float64}(undef, size(x))
    Threads.@threads for i = axes(x_plus,2)
        x_plus[:,i] = dynamics(Σ, x[:,i], u)
    end
    pf.particles = x_plus
end

## function: MeasurementUpdate
# input: 
#       - particle_filter: particle filter struct
#       - model: set of dynamics
#       - y: observation taken of true state
#       - var: measurement variance
#
# objective: calculate the likelihood associated with each particle in the density
function MeasurementUpdate(particle_filter, Σ, y)
    x = particle_filter.particles
    likelihoods = Vector(fill(0.0,(L)))
    Threads.@threads for i = axes(x,2)
        err = y-Σ.H(x[:,i])
        likelihoods[i] = exp.((-1/2)* err' * inv(Σ.R) *err)
    end
    particle_filter.likelihoods = particle_filter.likelihoods.*likelihoods./(sum(particle_filter.likelihoods.*likelihoods))
end


## function: Resampler
# input:
#   - particle_filter: particle filter objective
# objective: given updated particle likelihoods, resample from current particles to avoid depletion
function Resampler(particle_filter)
    x_resampled = fill(NaN, size(particle_filter.particles))
    CDF = cumsum(particle_filter.likelihoods)
    for i = axes(x_resampled,2)
        x_resampled[:,i] = particle_filter.particles[:,findfirst(CDF .>= rand(1))]
    end
    particle_filter.particles = x_resampled
    particle_filter.likelihoods = Vector(fill(1,(L)))
end

## functionL propagate_bootstrap_pf
# input: 
#       - pf: particle filter struct
#       - Σ: set of dynamics
#       - u: input signal
#       - y: measurement of true state to be used in boostrap pf
# 
# objective:
#       - propagate the particle density one time step based on the observation of the state
function propagate_bootstrap_pf(pf::Particle_Filter, Σ::Dynamics, u::Vector{Float64}, Ξ::Matrix{Float64}, y::Vector{Float64})
    pf.TimeUpdate(pf,Ξ, Σ, u)
    pf.MeasurementUpdate(pf, Σ, y)
    pf.Resampler(pf)
end