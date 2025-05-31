#
# Gradient Method - QP
# Lorenzo Sforni
# Bologna, 18/10/2022
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils, animation

seed = 42
np.random.seed(seed)

# implement of the aggregative tracking algorithm
max_iter = 500

def safety_controller(u_ref, adj, zz, agent_index, alpha):
    """
    Safety controller to ensure that the control input u_ref is safe with respect to the neighbors' positions zz.
    
    Parameters:
    u_ref : np.ndarray
        The reference control input.
    adj : np.ndarray
        The adjacency matrix representing the graph structure.
    zz : np.ndarray
        The positions of the agents.
    
    Returns:
    u_safe : np.ndarray
        The safe control input.
    """
    
    # Parameters
    gamma = 2/alpha  # Safety margin
    delta = 0.01  # safety distance
    # delta = d_min - 2 * alpha * v_max * eta  # safety distance, d_min is the minimum distance between agents, eta is a small perturbation to ensure safety
    # delta = 2 * alpha * v_max
    print(f"gamma: {gamma}, delta: {delta}")

    # Creation of the matrix A
    neighbors = np.where(adj > 0)[0]  # Get indices of neighbors
    neighbors = neighbors[neighbors != agent_index]  # Exclude the agent itself
    # print(f"Neighbors: {neighbors}")
    A = np.zeros((len(neighbors), u_ref.shape[0]))
    b = np.zeros(len(neighbors))
    # print(f"A shape: {A.shape}")
    # print(f"zz[i]: {zz[agent_index]}")
    for j in range(A.shape[0]):
        diff = (zz[agent_index] - zz[neighbors[j]])  # Difference between the agent's position and the neighbor's position
        # print(f"Difference for agent {agent_index} and neighbor {neighbors[j]}: {diff}")
        A[j] = -2 * diff.T
        b[j] = 0.5 * gamma *(np.linalg.norm(diff)**2 - delta**2)
        # b[j] = gamma *(np.linalg.norm(diff)**2 - delta**2)

    # print(f"A: {A}")
    # print(f"b: {b}")

    # Solve the QP problem
    QQ = np.eye(u_ref.shape[0]) # Identity matrix for the quadratic term    
    qq = - u_ref                # Linear term

    minimum, cost = min_cvx_solver(QQ, qq, A, b)
    print(f"Minimum: {minimum}, Cost: {cost}")
    
    return minimum, cost


def min_cvx_solver(QQ, qq, AA, bb):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/

    Obtain optimal solution for constrained QP

        min_{z} 1/2 z^T Q z + q^T z
        s.t.    Az - b <= 0

    """
    zz = cvx.Variable(qq.shape)

    # Cost function
    cost = 0.5* cvx.quad_form(zz,QQ) + qq.T @ zz

    # Constraint Az <= b
    constraint = [AA@zz <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    problem.solve()
    return zz.value, problem.value


def cost_fn(zz, rr, barycenter, gamma_1, gamma_2):
    """
    Compute the cost function and its gradient for the aggregative tracking problem.
    
    Parameters:
    zz : np.ndarray
        The positions of the agents at the current iteration.
    rr : np.ndarray
        The target positions of the agents.
    
    Returns:
    cost : float
        The computed cost value.
    grad : np.ndarray
        The gradient of the cost function with respect to zz.
    """
    # print(f"zz shape: {zz.shape}")
    # print(f"rr shape: {rr.shape}")
    # print(f"zz: {zz}")
    # print(f"rr: {rr}")
    target_norm = np.linalg.norm(zz-rr)
    # print(f"target_norm: {target_norm}")
    # print(f"target_norm shape: {target_norm.shape}")
    barycenter_norm = np.linalg.norm(zz - barycenter)
    # print(f"barycenter_norm: {barycenter_norm}")
    # print(f"barycenter_norm shape: {barycenter_norm.shape}")

    cost = gamma_1 * target_norm**2 + gamma_2 * barycenter_norm**2
    return cost


def aggregative_variable(zz): # barycenter of the agents' positions
    """
    Compute the aggregative variable (barycenter) of the agents' positions.
    Parameters:
    zz : np.ndarray
        The positions of the agents.
    Returns:
    barycenter : np.ndarray
        The barycenter of the agents' positions.
    """
    barycenter = np.mean(zz, axis=0)  # Compute the mean of the agents' positions
    return barycenter

## TODO: passare solo gamma_1 o gamma_2 in basa al tipo di gradiente
def gradient_computation(zz, rr, barycenter, gamma_1, gamma_2, N, type='first'):
    
    if type == 'first':
        # derivate of the cost function with respet to zz
        grad = 2 * gamma_1 * (zz - rr) + 2 * gamma_2 * (1 - 1/N) * (zz - barycenter)
    elif type == 'second':
        # derivate of the cost function with respect to sigma(z)
        grad = -2 * gamma_2 * (zz - barycenter)
    else:
        raise ValueError("Invalid type. Use 'first' or 'second'.")
    # print(f"grad shape: {grad.shape}")
    return grad

def aggregative_tracking(alpha, target_pos, z_init, adj, max_iter, N, d, cost_functions, gamma_1, gamma_2):
    # 3 states
    zz = np.zeros((max_iter, N, d))  # positions of the agents
    ss = np.zeros((max_iter, N, d))  # proxy of \sigma(x^k)
    vv = np.zeros((max_iter, N, d))  # proxy of \frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_J(z_j^k, \sigma(z^k))

    total_cost = np.zeros(max_iter)
    total_grad = np.zeros((max_iter, d))

    # TODO: would be cool to generalize the phi(z) func, e.g. to a weighted barycenter (VIP node) 
    # def phi(zz):
    #     zz

    # Initialization
    zz[0] = z_init 
    ss[0] = z_init # \phi_{i}(z) is the identity function
    for i in range(N):
        vv[0, i] = gradient_computation(zz[0,i], target_pos[i], ss[0, i], gamma_1[i], gamma_2[i], N, type='second')

    for k in range(max_iter-1):

        for i in range(N):
            # NOTE: usage of ss[k,i] instead of barycenter
            cost = cost_functions[i](zz[k,i], ss[k,i])
            
            # [ zz update ]
            nabla_1 = gradient_computation(zz[k,i], target_pos[i], ss[k,i], gamma_1[i], gamma_2[i], N, type='first')
            u_ref = nabla_1 + np.eye(d) @ vv[k,i]
            # u, _ = safety_controller(u_ref, adj[i], zz[k], i, alpha)  # Ensure safety with respect to neighbors
            zz[k+1, i] = zz[k, i] - alpha * u_ref

            # [ ss update ]
            # ss_k_T = np.moveaxis(ss[k], 0, -1) # from (N, d) to (d, N)
            ss_consensus = ss[k].T @ adj[i].T
            # ss_consensus = ss_consensus.squeeze()
            ss_local_innovation = zz[k+1, i] - zz[k, i]
            ss[k+1, i] = ss_consensus + ss_local_innovation

            # [ vv update ]
            # vv_k_T = np.moveaxis(vv[k], 0, -1) # from (N, d) to (d, N)
            vv_consensus = vv[k].T @ adj[i].T
            # vv_consensus = vv_consensus.squeeze()
            nabla2_k = gradient_computation(zz[k,i], target_pos[i], ss[k,i], gamma_1[i], gamma_2[i], N, type='second')
            nabla2_k_plus_1 = gradient_computation(zz[k+1,i], target_pos[i], ss[k+1,i], gamma_1[i], gamma_2[i], N, type='second')
            vv_local_innovation = nabla2_k_plus_1 - nabla2_k
            vv[k+1, i] = vv_consensus + vv_local_innovation

            total_cost[k] += cost
            total_grad[k] += u_ref
    
    return total_cost, total_grad, zz, ss, vv

if __name__ == "__main__":
    # setup 
    N = 3  # number of agents
    d = 2  # dimension of the state space

    # generate N target positions
    # target_pos = np.random.rand(N, d) * 10
    # target_pos = np.array([[10,10],[10,8], [10,6], [10,4], [10,2]])  # fixed initial positions for the agents
    target_pos = np.array([[2,9],[2,8], [2,10]])  # fixed initial positions for the agents
    
    print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))
    
    # generate initial positions for the agents
    # z_init = np.random.normal(size=(N, d)) * 10
    # z_init = np.array([[0,2],[0,4], [0,6], [0,8], [0,10]])  # fixed initial positions for the agents
    z_init = np.array([[0,8],[0,10],[0,6]])  # fixed initial positions for the agents
    print("Initial Positions: {}\tShape: {}".format(z_init, z_init.shape))

    # args = {'edge_probability': 0.65, 'seed': seed}
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.COMPLETE)

    # define ell_i
    gamma_1 = np.ones(N)  # equally distributed weights
    gamma_2 = np.ones(N)  # equally distributed weights
    # gamma_1 = np.random.uniform(0.1, 1.0, size=N)     # random weights for the cost functions
    # gamma_2 = np.random.uniform(0.1, 1.0, size=N)     # random weights for the cost functions
    
    cost_functions = []
    for i in range(N):
        cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))

    alpha = 0.1 # step size

    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    res = aggregative_tracking(alpha, target_pos, z_init, adj, max_iter, N, d, cost_functions, gamma_1, gamma_2)
    (total_cost_distr, total_grad_distr, zz_distr, ss_distr, vv_distr) = res

    # run simulations
    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

    ax = axes[0]
    # plot_utils.show_cost_evolution(ax, cost_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_cost_evolution(ax, total_cost_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    ax = axes[1]
    # plot_utils.show_norm_of_total_gradient(ax, grad_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    # ax = axes[2]
    # plot_utils.show_cost_evolution(ax, ss_distr[:,:,0], max_iter,semilogy=False, label="Distributed Aggregative Trackgin - vv")
    # plot_utils.show_consensus_error(ax, N, zz_centr, semilogy=False, label="Consensus Error Centralized")
    # plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    plot_utils.show_and_wait(fig)

    # fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    # animation.animation(ax, zz_centr, target_pos)

    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    animation.animation(ax, zz_distr, target_pos, adj)

    # fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    # animation.animation(ax, vv_distr, target_pos, adj)

    # print(f"vv_distr[450:max_iter]: {vv_distr[450:max_iter]}")
