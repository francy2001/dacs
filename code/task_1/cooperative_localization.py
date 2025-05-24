import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
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
plt.show()


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

# TODO: check con gradient centralized
# -------------------
# |   CENTRALIZED   |
# -------------------
# sum of the cost functions


# define ell_i
cost_functions = []
for i in range(N):
    # print(f"robot_pos[i]: {robot_pos[i]}")
    # print(f"robot_pos[i].shape: {robot_pos[i].shape}")
    # print(f"noisy_distances[i]: {noisy_distances[i]}")
    # print(f"noisy_distances[i].shape: {noisy_distances[i].shape}")
    cost_functions.append(lambda zz, i=i: cost_fn(zz, distances[i], robot_pos[i]))

# init s
for i in range(N):
    _, grad = cost_functions[i](z[0, i])
    s[0, i] = grad
    # print(f"s[0, {i}]: {s[0, i]}")
    # print(f"s[0, {i}].shape: {s[0, i].shape}")

# general case of a graph with birkhoff-von-neumann weights
graph, weighted_adj = gt.create_graph_birkhoff_von_neumann(N, 5)
gt.show_graph_and_adj_matrix(graph, weighted_adj)

cost, grad, zz, ss = gt.gradient_tracking(N, Nt, d, z, s, weighted_adj, cost_functions, alpha)

fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=3)

# fig.suptitle("Plot - Linear scale")
# fig.canvas.manager.set_window_title("Plot - Linear scale")

ax = axes[0]
# optimal cost error - one line! we are minimizing the sum not each l_i
ax.set_title("[Log] Cost")
ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1]))

ax = axes[1]
ax.set_title("[Normal] Cost")
ax.plot(np.arange(max_iter - 1), cost[:-1])
# ax.plot(np.arange(max_iter - 1), cost_opt * np.ones((max_iter - 1)), "r--")

ax  = axes[2]
ax.set_title("Norm of the total gradient")
total_grad = grad

print(f"total_grad[0:10]]: {total_grad[0:10]}")
print(f"total_grad.shape: {total_grad.shape}")

# total_grad_norm = np.linalg.norm(total_grad, axis=(1,2))
total_grad_norm = [np.linalg.norm(total_grad[k].flatten()) for k in range(max_iter)]
ax.plot(np.arange(max_iter - 1), total_grad_norm[:-1])

# ax.plot(np.arange(max_iter - 1), [) for k in range(max_iter - 1)])**2)

plt.show()




# Run multiple simulations
#   call gradient_tracking(N, list: QQ, list: rr)

# for each simulation:
# - comparison with the "centralized" version
# - plots: (log scale)
#    - evolution of the "cost" function
#    - norm of the TOTAL gradient of the cost function