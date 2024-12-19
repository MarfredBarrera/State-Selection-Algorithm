
MARKER_SIZE = 1.5
state_constraints = [
    rectangle_from_coords(x1_lowerlim,y1_lowerlim,x1_upperlim,y1_upperlim)
    rectangle_from_coords(x2_lowerlim,y2_lowerlim,x2_upperlim,y2_upperlim)]


function rectangle_from_coords(xb,yb,xt,yt)
    [
        xb  yb
        xt  yb
        xt  yt
        xb  yt
        xb  yb
        NaN NaN
    ]
end

function animate_frame(i)
    xlims!(-15,15)
    ylims!(-17,7)
    # plot the particle density
    plot!(sim_data[1,:,i],sim_data[2,:,i],seriestype=:scatter,
    label=false,
    ms=MARKER_SIZE,
    z_order=:back)

    # plot the candidate state
    scatter!((x_candidate[1,i], x_candidate[2,i]),
    label = false,
    mc =:red,
    z_order=:front)

    # plot the state constraints
    plot!(state_constraints[:,1], state_constraints[:,2],
        label = false,
        lc=:black)

end