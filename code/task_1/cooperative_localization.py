import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gradient_tracking as gt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils

seed = 38
np.random.seed(seed)

# Parameters
# Good practise: N >> Nt
Nt = 3  # number of targets
N = 8  # number of agents
max_iter = 5000

def cost_fn(zz, dd, pp):
    """
    Computes the cost and gradient for the distributed gradient tracking algorithm.
    
    Parameters:
    ----------
        zz: np.ndarray
            The state vector for the agent, shape (Nt, d).
        dd: np.ndarray
            The distance vector for the agent, shape (Nt,).
        pp: np.ndarray
            The position of the agent, shape (d,).
    
    Returns:
    -------
        cost: float
            The value of the cost function.
            l(zz) = \sum_{tau=1}^{Nt} (dd[tau]^2 - ||zz[tau] - pp||^2)^2
        grad: np.ndarray
            The gradient of the cost function with respect to zz, shape (Nt, d).
            grad(l(zz)) = -4 * (dd[tau]^2 - ||zz[tau] - pp||^2) * (zz[tau] - pp)
    """


    norms = np.linalg.norm(zz - pp, axis=1)
    D = dd**2 - norms**2
    # print(f"norms: {norms}, norms.shape: {norms.shape}")
    # print(f"D: {D}, D.shape: {D.shape}")    
    cost = D.T @ D

    grad = np.zeros((Nt, d))
    for tau in range(Nt):
        grad[tau] = -4 * D[tau] * (zz[tau] - pp)

    return cost, grad

# [ generate positions ]
d = 2           # positions are in R^2
dd = (Nt, d)    # state dimension

# [ random positions for robots ]
robot_pos = np.random.rand(N, d) * 10
print("Robot Positions: {}\tShape: {}".format(robot_pos, robot_pos.shape))

# [ random positions for targets ]
target_pos = np.random.rand(Nt, d) * 10
print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))

# [ calculate distances ]
distances = np.zeros((N, Nt))
for i in range(N):
    for j in range(Nt):
        distances[i,j] = np.linalg.norm(robot_pos[i] - target_pos[j])
print("Distances: {}\tShape: {}".format(distances, distances.shape))

# [ noisy distances ]
noise = np.random.normal(0, 0.1, distances.shape)
noisy_distances = distances + noise
print("Noisy Distances: {}".format(noisy_distances))

# [ plot robot and target positions ]
fig = plt.figure()
plt.xlabel('x')
plt.ylabel('y')
ax = fig.gca()
ax.plot(robot_pos[:,0], robot_pos[:,1], 'ro', label='Robot Positions')
ax.plot(target_pos[:,0], target_pos[:,1], 'bx', label='Target Positions')
plt.grid()
plot_utils.show_and_wait(fig)

# -------------------------------------
# |   DISTRIBUTED GRADIENT TRACKING   |
# -------------------------------------
alpha = 0.0003

# [ init z ]
z_init = np.random.uniform(low=-10, high=10, size=(N, Nt, d))
print("z_init: ", z_init)

# [ define ell_i ]
cost_functions = []
for i in range(N):
    # NOTE: noisy distances
    # cost_functions.append(lambda zz, i=i: cost_fn(zz, noisy_distances[i], robot_pos[i]))

    # NOTE: undistorted distances, show correctness of the algorithm
    cost_functions.append(lambda zz, i=i: cost_fn(zz, distances[i], robot_pos[i]))

# -----------------------------------------------------------------
# -------------------
# |   CENTRALIZED   | (Heavy-Ball method)
# -------------------
# sum of the cost functions, no need to use N (= number of agents)
def centralized_gradient_method(max_iter, dd, z_init, cost_functions):
    """
    Centralized gradient method using the Heavy-Ball method.

    Parameters:
    ----------
        max_iter: int
            The maximum number of iterations.
        dd: tuple
            The shape of the state vector (Nt, d).
        z_init: np.ndarray
            The initial state vector, shape (N, Nt, d).
        cost_functions: list
            A list of cost functions for each agent, where each function takes a state vector zz and returns a tuple (cost, gradient).
    
    Returns:
    -------
        cost: np.ndarray
            The cost at each iteration, shape (max_iter,).
        grad: np.ndarray
            The gradient at each iteration, shape (max_iter, Nt, d).
        zz: np.ndarray
            The state vector at each iteration, shape (max_iter, Nt, d).
        xi: np.ndarray
            The auxiliary state vector at each iteration, shape (max_iter, Nt, d).
        vv: np.ndarray
            The intermediate state vector at each iteration, shape (max_iter, Nt, d).
        yy: np.ndarray
            The update vector at each iteration, shape (max_iter, Nt, d).
    """

    # [ Heavy-Ball method ]
    # successive costs of "hb" i.e. heavy ball gradient method
    alpha1 = 0.0        # if == 0, it's the classic gradient method
    alpha2 = 0.00003
    alpha3 = 0.0

    def centralized_cost_fn(zz):
        total_cost = 0
        total_grad = np.zeros(shape=(dd)) # (Nt, d)
        for i in range(N):
            c, g = cost_functions[i](zz)
            total_cost += c
            total_grad += g

        return total_cost, total_grad
    
    # two states
    zz = np.zeros((max_iter, *dd)) # indeces: [time, which target, position-component]
    xi = np.zeros((max_iter, *dd))

    vv = np.zeros((max_iter, *dd))
    yy = np.zeros((max_iter, *dd))

    cost = np.zeros((max_iter))
    grad = np.zeros((max_iter, *dd))

    z_init_hb = np.mean(z_init, axis=0) # centralized mean of the random z_init
    zz[0] = z_init_hb
    xi[0] = z_init_hb

    for k in range(max_iter - 1):
        vv[k] = (1 + alpha3) * zz[k] - alpha3 * xi[k]
        cost[k], grad[k] = centralized_cost_fn(vv[k])

        yy[k] = -alpha2 * grad[k]
        
        zz[k + 1] = (1 + alpha1) * zz[k] - alpha1 * xi[k] + yy[k]
        xi[k + 1] = zz[k]
    return cost, grad, zz, xi, vv, yy

# [ run the centralized gradient method ]
cost_hb, grad_hb, zz_hb, _, _, _ = centralized_gradient_method(max_iter, dd, z_init, cost_functions)

# -----------------------------------------------------------------

# general case of a graph with birkhoff-von-neumann weights
args = {'edge_probability': 0.65, 'seed': seed}
graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)

# show graph and weighted adjacency matrix
fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
plot_utils.show_graph_and_adj_matrix(fig, axs, graph, weighted_adj)
plot_utils.show_and_wait(fig)

# run the gradient tracking method
dd = (Nt, d)
print(f"Running gradient tracking with alpha = {alpha:.6f}")
cost, grad, zz, ss = gt.gradient_tracking(max_iter, N, dd, z_init, weighted_adj, cost_functions, alpha)

fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

ax = axes[0]
# optimal cost error - one line! we are minimizing the sum not each l_i
plot_utils.show_cost_evolution(ax, cost, max_iter, semilogy=True, label="Distributed (GT)")
plot_utils.show_cost_evolution(ax, cost_hb, max_iter, semilogy=True, label="Centralized (HB)")

# ax = axes[1]
# plot_utils.show_cost_evolution(ax, cost, max_iter, semilogy=False, label="Distributed (GT)")
# plot_utils.show_cost_evolution(ax, cost_hb, max_iter, semilogy=False, label="Centralized (HB)")

ax  = axes[1]
total_grad = [grad[k].flatten() for k in range(max_iter)]
total_grad_hb = [grad_hb[k].flatten() for k in range(max_iter)]
plot_utils.show_norm_of_total_gradient(ax, total_grad, max_iter, semilogy=True, label="Distributed (GT)")
plot_utils.show_norm_of_total_gradient(ax, total_grad_hb, max_iter, semilogy=True, label="Centralized (HB)")

plot_utils.show_and_wait(fig)

# -------------------------
# |      SIMULATIONS      |
# -------------------------
run_simulations = True
p_ER = 0.65

if run_simulations:
    print("Simulations...")

    for graph_type in graph_utils.GraphType:
        # prepare args
        args = {}
        if graph_type == graph_utils.GraphType.ERDOS_RENYI:
            args = {
                'edge_probability': p_ER,
                'seed': seed
            }
        
        # create graph and show it
        graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_type, args)
        
        fig, axs = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
        title = f"Graph Type = {graph_type}"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        plot_utils.show_graph_and_adj_matrix(fig, axs[0], graph, weighted_adj)
        
        # [ run centralized ]
        cost_hb, grad_hb, zz_hb, _, _, _ = centralized_gradient_method(max_iter, dd, z_init, cost_functions)
        
        # [ run distributed ]
        cost, grad, zz, ss = gt.gradient_tracking(max_iter, N, dd, z_init, weighted_adj, cost_functions, alpha)

        total_grad = [grad[k].flatten() for k in range(max_iter)]
        plot_utils.show_cost_evolution(axs[1][0], cost, max_iter, semilogy=True, label="Distributed (GT)")
        plot_utils.show_cost_evolution(axs[1][0], cost_hb, max_iter, semilogy=True, label="Centralized (HB)")

        total_grad = [grad[k].flatten() for k in range(max_iter)]
        total_grad_hb = [grad_hb[k].flatten() for k in range(max_iter)]
        plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad, max_iter, semilogy=True, label="Distributed (GT)")
        plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad_hb, max_iter, semilogy=True, label="Centralized (HB)")

        plot_utils.show_and_wait(fig)