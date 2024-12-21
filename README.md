# State Selection Algorithm

This Julia program is an implementation of the State Selection Algorithm outlined in this paper: [State estimation for control: an approach for output-feedback stochastic MPC](https://arxiv.org/pdf/2303.00873).

We consider the discrete time, nonlinear dynamics of the numerical example in the paper and compare the performance of the State Selection Algorithm vs the nominal conditional mean method of state selection. 

The novelty of the State Selection Algorithm is that, through a Nested Monte Carlo simulation structure, we can "try out" every particle in the density and select a feasible state with minimal cost to be used in a feedback controller intead of conditioning the control signal itself. This differs from a the traditional method of state estimation, which simply selects the conditional mean of the particle density. 

In the case of the nonlinear dynamics and state constraints specified in the paper, the State Selection Algorithm will select states closer to the contraint boundaries, which is a result of the prescribed cost function and feedback controller.

### Discete Time Dynamics:
$x_k = [z_k,h_k]^T$

$z_{k+1} = 0.9z_k + 0.2h_k + w_1$

$h_{k+1} = -0.15z_k + 0.9h_k + 0.05z_kh_k + u_k + w_2 $

### Feedback Controller:
$u_k = -0.05z_kh_k$

### Running Cost:
$l_k = x_k^Tx_k + u_k^2$

### Constraint Set:

$L = [3,5] \times[-4,2]\:\:  \cup \:\: [-2,5] \times [-7,-4]$

## SSA Plot
![](/Saved_plots/ssa_gif_example.gif)

## Conditional Mean Plot
![](/Saved_plots/cm_example.gif)

## Violation Rates
![](/Saved_plots/violation_rates.png)

In the case of these dynamics, the minimum cost trajectories that do not violate the state constraints push the density away from the origin. Eventually, the particles must stabilize near the origin. This behavior is evident in the differences in propagation between the SSA and Conditional Mean particle filters.


### Notes:
1) Theoretically, every step of the Nested Monte Carlo simulations are independent and can be completely parallelized with GPU programming. This Julia program is partially GPU parallelized, but further work can be done to completely parallelize the simulations. 

2) Currently, the program cannot simultaneously run the calculations and plot the result, so they need to be run one after the other. In the case that you see an error message saying "__data is undefined", it is likely that there is no simulation data stored, so you must run a simulation before plotting it. 
