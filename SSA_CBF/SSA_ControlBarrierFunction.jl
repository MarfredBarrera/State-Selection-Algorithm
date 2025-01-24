"""
    ControlBarrierFunction

Control barrier function (CBF) defining a safe set as its zero superlevel set.

# Fields
- `h::Function` : function defining the safe set `h(x) ≥ 0`
- `α::Function` : extended class K function `a(h(x))` for CBF
- `∇h::Function` : gradient of CBF
- `Lfh::Function` : Lie derivative of CBF along drift vector field `f`
- `Lgh::Function` : Lie derivative of CBF along control directions `g`
"""
struct ControlBarrierFunction
    h::Function
    α::Function
    ∇h::Function
    Lfh::Function
    Lgh::Function
end

function ControlBarrierFunction(h::Function, Σ::Dynamics, α::Function)
    ∇h(x) = Σ.n == 1 ? ForwardDiff.derivative(h, x) : ForwardDiff.gradient(h, x)
    Lfh(x) = ∇h(x)' * Σ.f(x)
    Lgh(x) = ∇h(x)' * Σ.g(x)

    return ControlBarrierFunction(h, α, ∇h, Lfh, Lgh)
end

(cbf::ControlBarrierFunction)(x) = cbf.h(x)

