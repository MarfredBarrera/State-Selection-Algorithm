include("SSA_dynamics.jl")
using LinearAlgebra

mutable struct Particle_Filter
    model::Model
    TimeUpdate::Function
    MeasurementUpdate::Function
    Resampler::Function
    likelihoods::Vector
    particles::Array
end

function TimeUpdate(x, model, u, w)
    x_plus = Array{Float64}(undef, n, size(x,2))
    for i = axes(x,2)
        x_plus[:,i] = model.f(x[:,i],u,w[:,i])
    end
    return x_plus
end

function MeasurementUpdate!(particle_filter, model, y)
    x = particle_filter.particles
    measurement_error = y.-model.h(x,0)
    likelihoods = exp.((-1/2)*(measurement_error.^2)*inv(model.R))
    particle_filter.likelihoods = particle_filter.likelihoods.*likelihoods./(sum(particle_filter.likelihoods.*likelihoods))
end


function Resampler(particle_filter)
    x_resampled = fill(NaN, size(particle_filter.particles))
    CDF = cumsum(particle_filter.likelihoods)
    for i = 1:length(particle_filter.likelihoods)
        x_resampled[:,i] = particle_filter.particles[:,findfirst(CDF .>= rand(1))]
    end
    particle_filter.particles = x_resampled
    particle_filter.likelihoods = Vector(fill(1,(L)))
end