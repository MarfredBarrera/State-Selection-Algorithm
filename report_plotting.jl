using Serialization

(x_star_queue, particle_queue, violation_queue, cost_queue) = deserialize("simulation_results_with_SSA.jls")
(x_star_queue_noSSA, particle_queue_noSSA, violation_queue_noSSA, cost_queue_noSSA) = deserialize("simulation_results_without_SSA.jls")
r0 = [1.85, 1.55]
dmin = 1.25
T = 70



function plot_result(particle_queue, x_star_queue, r0, dmin, T, run_ssa)

    # intialize plot
    plot(particle_queue[3][1,:], particle_queue[3][2,:],
            seriestype=:scatter, 
            ms = 1.2, 
            label = false, 
            z_order=:front)

    plot!((x_star_queue[3][1], x_star_queue[3][2]), 
        seriestype=:scatter, 
        ms = 3.50, 
        label = L"Candidate State $x^{\star}$",  # Add label only for the first iteration
        legend = :topleft, 
        mc = :blue, 
        z_order=:front)

    for (i, t) in enumerate(1:5:T-3)
        plot!(particle_queue[t][1,:], particle_queue[t][2,:],
                seriestype=:scatter, 
                ms = 1.2, 
                label = false,
                z_order=:front)
        plot!((x_star_queue[t][1], x_star_queue[t][2]), 
                seriestype=:scatter, 
                ms = 3.50, 
                legend = :topleft, 
                label = false,
                mc = :blue, 
                z_order=:front)
    end

    plot!(circleShape(r0[1],r0[2],dmin), seriestype = [:shape,], lw=0.5,
    linecolor =:black, ls=:dash, c =:red,
    label = false, aspect_ratio =1,
    z_order=:back)
    plot!(circleShape(0,0,0.25), seriestype = [:shape,], lw=0.5,
        c =:green, linecolor =:black,
        label = false, aspect_ratio =1, z_order=:back)
    xlims!(-1.5, 5.0)
    ylims!(-1.5, 5.0)

    if(run_ssa)
        savefig("SSA.png")
    else
        savefig("noSSA.png")
    end
end

plot_result(particle_queue, x_star_queue, r0, dmin, T, true)
plot_result(particle_queue_noSSA, x_star_queue_noSSA, r0, dmin, T, false)