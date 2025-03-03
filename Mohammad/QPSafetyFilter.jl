"""
Controller that solves a control barrier function-based quadratic program (CBF-QP).

Uses OSQP to solve the corresponding QP.

# Fields
- `k::Function` : function that computes safe control actions
"""

struct QPSafetyFilter
    k::Function
end

"""
    (k::QPSafetyFilter)(x)

Functors for evaluating QP-based safety filter
"""
(k::QPSafetyFilter)(x) = k.k(x)

# Construct a QPSafetyFilter from a cbf, dynamical system, and nominal control function
function QPSafetyFilter(
    cbf::ControlBarrierFunction, Σ::ControlAffineDynamics, kd::Function
)
    return QPSafetyFilter(x -> solve_cbf_qp(x, Σ::ControlAffineDynamics, cbf::ControlBarrierFunction, kd::Function))
end


"""
solver for CBF_QP quadratic program safety filter given a CBF and nominal controller
"""
function solve_cbf_qp(x, Σ::ControlAffineDynamics, cbf::ControlBarrierFunction, kd::Function)
    c1 = 1.0
    c2 = 1.0
    u_nom = kd(x)
    model = Model(OSQP.Optimizer)
    set_silent(model)
    u = Σ.m == 1 ? @variable(model, u) : @variable(model, u[1:(Σ.m)])

    @objective(model, Min, sum((u[i]-u_nom[i])^2 for i in 1:Σ.m))
    @constraint(model, (cbf.∇h(x)'*Σ.A*Σ.B + c1*cbf.∇h(x)'*Σ.g(x))*u >= -(cbf.∇h(x)'*Σ.A*(Σ.f(x)) + c1*cbf.∇h(x)'*Σ.f(x) + c2*cbf.h(x)))

    optimize!(model)

    return Σ.m == 1 ? value(u) : value.(u)
end