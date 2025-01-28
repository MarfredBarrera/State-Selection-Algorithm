include("SSA_CBF_main.jl")


function circleShape(h,k,r)
    θ = LinRange(0,2*pi,500)
    h .+ r*sin.(θ), k.+r*cos.(θ)
end


plot(circleShape(r_goal[1],r_goal[2],0.1), seriestype = [:shape,], lw=0.5,
c =:green, linecolor =:black,
legend = false, aspect_ratio =1)
plot!(circleShape(r0[1],r0[2],dmin), seriestype = [:shape,], lw=0.5,
     linecolor =:black, ls=:dash, c =:red,
     legend = false, aspect_ratio =1)

anim = @animate for t = 1:N
     plot(state[1,t,:], state[2,t,:], seriestype=:scatter, ms = 1.0, label = false, mc =:black, z_order=:front)
     plot!((xtrue[1,t], xtrue[2,t]), seriestype=:scatter, ms = 2.5, label=false, mc=:blue, z_order=:front)
     plot!(circleShape(r0[1],r0[2],dmin), seriestype = [:shape,], lw=0.5,
     linecolor =:black, ls=:dash, c =:red,
     legend = false, aspect_ratio =1,
     z_order=:back)

     xlims!(-1, 5)
     ylims!(-1, 5)
end
# savefig("HCW_CBF.png")

gif(anim, "HCW_CBF.gif",fps=10)