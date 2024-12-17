include("SSA_utils.jl")
include("SSA_particle_filter.jl")
include("SSA_dynamics.jl")
include("SSA_kernels.jl")

using Plots

struct Params
    M :: Int64
    N :: Int64
    L :: Int64
    n :: Int64
end

struct Gaussians
    # state density with mean and variance
    μ :: Float64
    Σ :: Float64

    # process noise ωₖ and scalar measurement noise vₖ
    ω :: Float64
    v :: Float64
end

struct Limits
    Ulim :: Float64
    α :: Float32
    ϵ :: Float32
    δ :: Float64

    x1_upperlim :: Float64
    x1_lowerlim :: Float64
    y1_upperlim :: Float64
    y1_lowerlim :: Float64

    x2_upperlim :: Float64
    x2_lowerlim :: Float64
    y2_upperlim :: Float64
    y2_lowerlim :: Float64
end

# intialize parameters
M = 1000
N = 5
L = 3000
n = 2
T = 10

# state density mean and variance
μ = 7.5 
Σ = 0.5

#  process noise variance ωₖ and scalar measurement noise variance vₖ
ω = 0.3
v = 0.3

Ulim = 3
α = 0.10
ϵ = 0.20
δ = 0.001

# state constraints
x1_upperlim = 5
x1_lowerlim = 3
y1_upperlim = 2
y1_lowerlim = -4

x2_upperlim = 5
x2_lowerlim = -2
y2_upperlim = -4
y2_lowerlim = - 7
 

# store parameters in struct
SSA_params = Params(M, N, L, n)
SSA_gauss  = Gaussians(μ, Σ, ω, v)
SSA_limits = Limits(Ulim, α, ϵ, δ, 
    x1_upperlim, x1_lowerlim, 
    y1_upperlim, y1_lowerlim,
    x2_upperlim, x2_lowerlim, 
    y2_upperlim, y2_lowerlim)



# initialize dynamics
Q = Matrix{Float64}(I, 2, 2)
R = v
dynamics = Model(f,h,u,Q,R)
x_true = Array{Float64}(undef, n, T)

# generate state density Xi according to Gaussian parameters
(Ξ_gaussian, W_gaussian, V_gaussian) = init_gaussians()
Ξ = gpu_generate_Xi(SSA_params.L, SSA_params.n)

# initialize particle filter
likelihoods = Vector(fill(1,(L)))
pf = Particle_Filter(dynamics, TimeUpdate, MeasurementUpdate!, Resampler, likelihoods, Array(Ξ))

x_true[:,1] = [μ; -μ].+ sqrt(Σ)*randn(2)

# run the state selection algorithm for the particle density
CUDA.@sync candidate_state, u_star, candidate_index = state_selection_algorithm(Ξ,SSA_params,SSA_limits)


### run bootstrap particle filter
Ξ = Array(Ξ)
w = Array(gpu_sample_gaussian_distribution(0, ω, (n,L,1)))
w_true = sqrt(ω)*randn(2)

# propagate particle density
Xi_plus = pf.TimeUpdate(Ξ, dynamics, u_star,w)
pf.particles = Xi_plus
x_true[:,2] = dynamics.f(x_true[:,1], u_star, w_true)

# take measurement
y = dynamics.h(x_true[:,2], sqrt(v)*randn())

# calculate likelihoods
pf.MeasurementUpdate(pf,dynamics,y)

pf.Resampler(pf)

plot(Ξ[1,:],Ξ[2,:], seriestype=:scatter)
plot!(pf.particles[1,:], pf.particles[2,:], seriestype=:scatter)
plot!(x_true[1,:], x_true[2,:], seriestype=:scatter)
xlims!(-15,15)
ylims!(-17,7)
