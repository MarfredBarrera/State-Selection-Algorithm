include("main.jl")

using LaTeXStrings


function circleShape(h,k,r)
    θ = LinRange(0,2*pi,500)
    h .+ r*sin.(θ), k.+r*cos.(θ)
end
function animate_gif(particle_queue, x_star_queue, r0, dmin, T, run_ssa)
    anim = @animate for t = 1:T-1
         plot(particle_queue[t][1,:], particle_queue[t][2,:],
              seriestype=:scatter, 
              ms = 1.0, 
              label = false, 
              mc =:black, 
              z_order=:front)
         plot!((x_star_queue[t][1], x_star_queue[t][2]), 
              seriestype=:scatter, 
              ms = 3.50, 
              label=L"Candidate State $x^{\star}$",
              legend =:topleft, 
              mc=:blue, 
              z_order=:front)
         plot!(circleShape(r0[1],r0[2],dmin), seriestype = [:shape,], lw=0.5,
              linecolor =:black, ls=:dash, c =:red,
              label = false, aspect_ratio =1,
              z_order=:back)
         plot!(circleShape(0,0,0.25), seriestype = [:shape,], lw=0.5,
              c =:green, linecolor =:black,
              label = false, aspect_ratio =1, z_order=:back)
         xlims!(-1.5, 5.0)
         ylims!(-1.5, 5.0)
    end

    if(run_ssa)
         gif(anim, "SSA.gif",fps=10)
    else
         gif(anim, "noSSA.gif",fps=10)
    end
end

animate_gif(particle_queue, x_star_queue, r0, dmin, T, run_ssa)
animate_gif(particle_queue_noSSA, x_star_queue_noSSA, r0, dmin, T, !run_ssa)


plot(1:T, violation_queue, label="SSA", xlabel="Time Step", ylabel="Constraint Violation Rate", lw=2)
plot!(1:T, violation_queue_noSSA, label="No SSA", lw=2)
plot!([0, T], [α, α], label="Constraint Violation Limit, α", lw=2, ls=:dash, c=:black)
savefig("constraint_violation.png")

plot(1:T, cost_queue, label="SSA", xlabel="Time Step", ylabel="Cost", lw=2)
plot!(1:T, cost_queue_noSSA, label="No SSA", lw=2)
savefig("cost.png")