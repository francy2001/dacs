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

# parameters
# TODO: check vincolo che N >> Nt
Nt = 6
# N = 3 * Nt
N = 17
max_iter = 10000

def cost_fn(zz, dd, pp):

    # TODO: check che vada bene il calcolo del cost e del gradiente
    # print("zz.shape: ", zz.shape)
    # print("pp: ", pp)
    # print("zz-pp: ", zz - pp)
    # print("zz-pp shape: ", (zz-pp).shape)
    norms = np.linalg.norm(zz - pp, axis=1)
    # print("norms: ", norms)
    # print("norms shape: ", norms.shape)
    # print("dd.shape: ", dd.shape) 
    # print("dd**2: ", dd**2)
    D = dd**2 - norms**2

    cost = D.T @ D
    # print("cost shape: ", cost.shape)
    grad = np.zeros((Nt, d))
    
    for tau in range(Nt):
        # grad[tau] = -4 * ((dd[tau]**2 - norms[tau]**2) * (zz[tau] - pp))
        grad[tau] = -4 * D[tau] * (zz[tau] - pp)

    # NOTE: this works but it changes the algorithms and the convergence rate guarantees
    # Normalization of the gradient due to the exploding gradient
    # epsilon = 1e-8  # avoid division by zero
    # for tau in range(Nt):
    #     raw_grad = -4 * ((dd[tau]**2 - norms[tau]**2) * (zz[tau] - pp))
    #     norm = np.linalg.norm(raw_grad)
    #     if norm > 0:
    #         grad[tau] = raw_grad / (norm + epsilon)
    #     else:
    #         grad[tau] = raw_grad  # o np.zeros_like(raw_grad)

    return cost, grad

# [ generate positions ]
# TODO: generate not overlapping positions (cfr: Feistel network for a discretized spawn map)
d = 2           # positions are in R^2
dd = (Nt, d)    # state dimension

robot_pos = np.random.rand(N, d) * 10
print("Robot Positions: {}\tShape: {}".format(robot_pos, robot_pos.shape))

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
## chosing alpha
alpha = 0.0003
# alpha_init = 0.000003
# alpha_step = 0.000050
# growth_rate = 1.1
# alpha = alpha_init - (alpha_step * growth_rate**Nt)

# init z
# TODO: educated guess? maybe choose the starting position of the target 
# on a sphere of radius noisy_distances[i] centered in robot_pos[i].
# z_init = np.random.uniform(size=(N, Nt, d))
z_init = np.random.uniform(low=-10, high=10, size=(N, Nt, d))
print("z_init: ", z_init)

# define ell_i
cost_functions = []
for i in range(N):
    # print(f"robot_pos[i]: {robot_pos[i]}")
    # print(f"robot_pos[i].shape: {robot_pos[i].shape}")
    # print(f"noisy_distances[i]: {noisy_distances[i]}")
    # print(f"noisy_distances[i].shape: {noisy_distances[i].shape}")
    
    # NOTE: noisy distances
    # cost_functions.append(lambda zz, i=i: cost_fn(zz, noisy_distances[i], robot_pos[i]))

    # NOTE: undistorted distances, show correctness of the algorithm
    cost_functions.append(lambda zz, i=i: cost_fn(zz, distances[i], robot_pos[i]))

# --------------------------------------------------------------
# TODO: check con gradient centralized
# -------------------
# |   CENTRALIZED   | (Heavy-Ball method)
# -------------------
# sum of the cost functions, no need to use N (= number of agents)

def centralized_gradient_method(max_iter, dd, z_init, cost_functions):
    # pdf: [ Optimization Basics ]
    # slide: [ 26/27 ] - "Accelerated gradient method: the heavy-ball method"
    # -----------------------------
    # |     HEAVY BALL METHOD     |
    # -----------------------------
    # successive costs of "hb" i.e. heavy ball gradient method
    alpha1 = 0.0 # if == 0, it's the classic gradient method
    alpha2 = 0.00003
    # alpha2 = alpha
    alpha3 = 0.0

    # print(f"Running gradient tracking with alpha = {alpha2:.6f}")

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
# graph, weighted_adj = graph_utils.create_graph_birkhoff_von_neumann(N, 5)
graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.COMPLETE)

# show graph and weighted adjacency matrix
fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
plot_utils.show_graph_and_adj_matrix(fig, axs, graph, weighted_adj)
plot_utils.show_and_wait(fig)



# run the gradient tracking method
dd = (Nt, d)
print(f"Running gradient tracking with alpha = {alpha:.6f}")
cost, grad, zz, ss = gt.gradient_tracking(max_iter, N, dd, z_init, weighted_adj, cost_functions, alpha)

fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=3)

ax = axes[0]
# optimal cost error - one line! we are minimizing the sum not each l_i
plot_utils.show_cost_evolution(ax, cost, max_iter, semilogy=True, label="Distributed (GT)")
plot_utils.show_cost_evolution(ax, cost_hb, max_iter, semilogy=True, label="Centralized (HB)")

ax = axes[1]
plot_utils.show_cost_evolution(ax, cost, max_iter, semilogy=False, label="Distributed (GT)")
plot_utils.show_cost_evolution(ax, cost_hb, max_iter, semilogy=False, label="Centralized (HB)")

ax  = axes[2]
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
        plot_utils.show_cost_evolution(axs[1][0], cost, max_iter, semilogy=False, label="Distributed (GT)")
        plot_utils.show_cost_evolution(axs[1][0], cost_hb, max_iter, semilogy=False, label="Centralized (HB)")

        total_grad = [grad[k].flatten() for k in range(max_iter)]
        total_grad_hb = [grad_hb[k].flatten() for k in range(max_iter)]
        plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad, max_iter, semilogy=True, label="Distributed (GT)")
        plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad_hb, max_iter, semilogy=True, label="Centralized (HB)")

        plot_utils.show_and_wait(fig)



# Run multiple simulations
#   call gradient_tracking(N, list: QQ, list: rr)

# for each simulation:
# - comparison with the "centralized" version
# - plots: (log scale)
#    - evolution of the "cost" function
#    - norm of the TOTAL gradient of the cost function