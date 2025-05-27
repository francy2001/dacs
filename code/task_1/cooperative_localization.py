import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graph_utils
import plot_utils
import gradient_tracking as gt

seed = 42
np.random.seed(seed)

# parameters
# TODO: check vincolo che N >> Nt
Nt = 3
N = 3 * Nt
max_iter = 3000

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
        grad[tau] = -4 * ((dd[tau]**2 - norms[tau]**2) * (zz[tau] - pp))

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

# generate positions
d = 2
# robot positions
robot_pos = np.random.rand(N, d) * 10
# TODO: posizioni che non si sovrappongono (guarda: rete di Feistel)
print("Robot Positions: {}\tShape: {}".format(robot_pos, robot_pos.shape))
# target positions
target_pos = np.random.rand(Nt, d) * 10
print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))
# generate distances
distances = np.zeros((N, Nt))
for i in range(N):
    for j in range(Nt):
        distances[i,j] = np.linalg.norm(robot_pos[i] - target_pos[j])
print("Distances: {}\tShape: {}".format(distances, distances.shape))
# generate noisy distances
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
#
# alphas = np.logspace(-4, -3, num=10) --> to have a set of possible values and chose the best one
#
# alpha = 1e-4 --> 3000 iterazioni arriva a poco più di 1e-9
# alpha = 0.0001291549665014884 --> 3000 iterazioni arriva a poco più di 1e-12
# alpha = 0.0001668100537200059 --> 3000 iterazioni arriva a poco più di 1e-17
# alpha = 0.00021544346900318845 --> 3000 iterazioni arriva a poco più di 1e-21 e si stabilizza (best value!!!)
# alpha = 0.0002782559402207126 --> 3000 iterazioni arriva a poco più di 1e-21 e si stabilizza (best value!!!)
# alpha = 0.0003593813663804626 --> 3000 iterazioni arriva a poco più di 1e-21 e si stabilizza (best value!!!)
# alpha = 0.0004661588769516728 --> esplode

alpha = 0.0003593813663804626    

# two states
z = np.zeros((max_iter, N, Nt, d)) # indeces: [time, who, which target, position-component]
s = np.zeros((max_iter, N, Nt, d))

# init z
z_init = np.random.normal(size=(N, Nt, d))
z[0] = z_init
print(f"z_init: {z_init}")
print(f"z_init shape: {z_init.shape}")


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
# sum of the cost functions

def centralized_cost_fn(zz):
    total_cost = 0
    total_grad = np.zeros(shape=(Nt, d))
    for i in range(N):
        c, g = cost_functions[i](zz)
        total_cost += c
        total_grad += g

    return total_cost, total_grad


# two states
z_hb = np.zeros((max_iter, Nt, d)) # indeces: [time, which target, position-component]
xi_hb = np.zeros((max_iter, Nt, d))

v = np.zeros((max_iter, Nt, d))
y = np.zeros((max_iter, Nt, d))

# successive costs of "hb" i.e. heavy ball gradient method
cost_hb = np.zeros((max_iter))
grad_hb = np.zeros((max_iter, Nt, d))

z_init_hb = np.mean(z_init, axis=0) # centralized mean of the random z_init
z_hb[0] = z_init_hb
xi_hb[0] = z_init_hb

alpha1 = 0.0 # if == 0, it's the classic gradient method
alpha2 = alpha
alpha3 = 0.0

# pdf: [ Optimization Basics ]
# slide: [ 26/27 ] - "Accelerated gradient method: the heavy-ball method"
# -----------------------------
# |     HEAVY BALL METHOD     |
# -----------------------------
for k in range(max_iter - 1):
    v[k] = (1 + alpha3) * z_hb[k] - alpha3 * xi_hb[k]
    cost_hb[k], grad_hb[k] = centralized_cost_fn(v[k])

    y[k] = -alpha2 * grad_hb[k]
    
    z_hb[k + 1] = (1 + alpha1) * z_hb[k] - alpha1 * xi_hb[k] + y[k]
    xi_hb[k + 1] = z_hb[k]

# -----------------------------------------------------------------

# init s
for i in range(N):
    _, grad = cost_functions[i](z[0, i])
    s[0, i] = grad
    # print(f"s[0, {i}]: {s[0, i]}")
    # print(f"s[0, {i}].shape: {s[0, i].shape}")

# general case of a graph with birkhoff-von-neumann weights
graph, weighted_adj = graph_utils.create_graph_birkhoff_von_neumann(N, 5)

# show graph and weighted adjacency matrix
fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
plot_utils.show_graph_and_adj_matrix(fig, axs, graph, weighted_adj)
plot_utils.show_and_wait(fig)

# run the gradient tracking method
cost, grad, zz, ss = gt.gradient_tracking(N, Nt, d, z, s, weighted_adj, cost_functions, alpha)

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
simulation = True
p_ER = 0.65

if simulation:
    print("Simulations...")

    for graph_type in graph_utils.GraphType:
        # TODO: for each one do the comparison with the centralized version
        args = {}

        # prepare args
        if graph_type == graph_utils.GraphType.ERDOS_RENYI:
            args = {
                'edge_probability': p_ER,
                'seed': seed
            }
        
        # create graph and show it
        graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_type, args)
        
        fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=2)
        title = f"Graph Type = {graph_type}"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        plot_utils.show_graph_and_adj_matrix(fig, axs[0], graph, weighted_adj)
        
        # run gradient tracking with the previously defined cost functions
        # TODO: is it right to reuse the same z and s?
        cost, grad, zz, ss = gt.gradient_tracking(N, Nt, d, z, s, weighted_adj, cost_functions, alpha)

        total_grad = [grad[k].flatten() for k in range(max_iter)]
        plot_utils.show_cost_evolution(axs[1][0], cost, max_iter, semilogy=True)
        plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad, max_iter, semilogy=True)
        
        plot_utils.show_and_wait(fig)



# Run multiple simulations
#   call gradient_tracking(N, list: QQ, list: rr)

# for each simulation:
# - comparison with the "centralized" version
# - plots: (log scale)
#    - evolution of the "cost" function
#    - norm of the TOTAL gradient of the cost function