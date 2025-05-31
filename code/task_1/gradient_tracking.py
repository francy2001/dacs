import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils

seed = 48
np.random.seed(seed)

max_iter = 1000
run_simulations = False

def __random_rotation_matrix(N):
    # Step 1: Generate a random N x N matrix
    A = np.random.rand(N, N)
    
    # Step 2: Perform QR decomposition
    # Compute the qr factorization of a matrix. Factor the matrix a as qr, where q is orthonormal and r is upper-triangular.
    Q, R = np.linalg.qr(A)
    
    # Step 3: Ensure a proper rotation matrix (det(Q) should be 1)
    # If the determinant is negative, flip the sign of the last column
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    
    return Q

def quadratic_cost_fn(zz, QQ, rr):
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(max_iter, NN, dd: tuple, zz_init, weighted_adj, cost_functions, alpha):
    """
    Gradient tracking.
    
    Parameters:
    ----------
        dd: tuple
            for multidimensional state, e.g. dd=(Nt, d), each agents has a state containing Nt positions in R^d, one for each target
    
    """

    # Task 1.1: (max_iter, NN, d)     --> [time, who, position-component]
    # Task 1.2: (max_iter, NN, Nt, d) --> [time, who, which target, position-component]
    zz = np.zeros((max_iter, NN, *dd))
    ss = np.zeros((max_iter, NN, *dd))

    # init z
    zz[0] = zz_init

    # init s
    for i in range(NN):
        _, grad = cost_functions[i](zz[0, i])
        ss[0, i] = grad

    cost = np.zeros((max_iter))
    grad = np.zeros((max_iter, *dd))

    for k in range(max_iter - 1):
        # print(f"ss[{k}]: {ss[k]}")
        # print(f"ss[{k}].shape: {ss[k].shape}")

        for i in range(NN):
            # print(f"zz[k].shape: {zz[k].shape}")
            # print(f"zz[k][0][0]: {zz[k][0][0]}")
            # print(f"np.transpose(zz[k]).shape: {np.moveaxis(zz[k], 0, -1).shape}")
            # print(f"ss[k,i].shape: {ss[k,i].shape}")
            # print(f"weighted_adj[i].T.shape: {weighted_adj[i].T.shape}")

            # NOTE: we have a zz[k].shape: (N, *state_dim), we would like to multiply with a (N,) vector
            # Let's move the agent-axis in the last index such that we have: (*state_dim, N)@(N,1)-->(*state_dim,1)
            # If state_dim is only a tuple containing (d,) then it's equivalent to take the transpose.
            zz_k_T = np.moveaxis(zz[k], 0, -1)
            # print(f"zz_k_T : {zz_k_T}")
            # print(f"zz[k].T : {zz[k].T}")

            zz[k+1, i] = zz_k_T @ weighted_adj[i].T - alpha * ss[k, i]

            # Move the agent-axis in the last index such that we have (Nt,d,N)@(N,1)-->(Nt,d,1)
            ss_k_T = np.moveaxis(ss[k], 0, -1)
            consensus = ss_k_T @ weighted_adj[i].T
            
            cost_k_i, grad_k_i = cost_functions[i](zz[k, i])
            _, grad_k_plus_1_i = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_k_plus_1_i - grad_k_i
            ss[k+1, i] = consensus + local_innovation

            cost[k] += cost_k_i
            grad[k] += grad_k_i # total gradient, centralized

        # grad[k] = sum of NN matrices with shape (Nt x d)

    return cost, grad, zz, ss


if __name__ == "__main__":
    # TODO: parametri a linea di comando
    NN = 10     # number of agents
    d = 2       # dimension of the state
    p_ER = 0.65 

    QQ_list = []
    rr_list = []

    for i in range(NN):
        Q = np.diag(np.random.uniform(size=d)) # TODO: use "SVD", syntetize rotations
        # R = __random_rotation_matrix(d) # NOTE: it's not working properly...
        # Q = R @ Q

        r = np.random.uniform(size=d)

        QQ_list.append(Q)
        rr_list.append(r)

    cost_functions = []
    for i in range(NN):
        cost_functions.append(lambda zz, i=i: quadratic_cost_fn(zz, QQ_list[i], rr_list[i]))

    # -------------------
    # |   CENTRALIZED   |
    # -------------------
    # I have the sum of all the costs functions ell_i.
    QQ_centralized = sum(QQ_list)
    rr_centralized = sum(rr_list)

    z_opt = -np.linalg.inv(QQ_centralized) @ rr_centralized
    print(f"[centr] z_opt: {z_opt}")

    cost_opt, _ = quadratic_cost_fn(z_opt, QQ_centralized, rr_centralized)
    print(f"[centr] cost_opt: {cost_opt}")
    
    # -------------------------------------
    # |   DISTRIBUTED GRADIENT TRACKING   |
    # -------------------------------------
    alpha = 0.05

    # create graph
    args = {
        'edge_probability': p_ER,
        'seed': seed
    }
    graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(NN, graph_utils.GraphType.ERDOS_RENYI, args)
    
    # show graph and adj matrix
    fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
    plot_utils.show_graph_and_adj_matrix(fig, axs, graph, weighted_adj)
    plot_utils.show_and_wait(fig)

    # run gradient tracking
    dd = (d,)
    zz_init = np.random.normal(size=(NN, d))
    cost, grad, zz, ss = gradient_tracking(max_iter, NN, dd, zz_init, weighted_adj, cost_functions, alpha)
    
    def show_plots(semilogy):
        fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=4)
        title = f"Plots {'(Logaritmic y scale)' if semilogy else '(Linear scale)'}"
        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)
        plot_utils.show_cost_evolution(axes[0], cost, max_iter, semilogy, cost_opt)
        plot_utils.show_optimal_cost_error(axes[1], cost, cost_opt, max_iter, semilogy)
        plot_utils.show_consensus_error(axes[2], NN, zz, max_iter, semilogy)
        plot_utils.show_norm_of_total_gradient(axes[3], grad, max_iter, semilogy)
        plot_utils.show_and_wait(fig)

    # show_plots(semilogy=False)
    show_plots(semilogy=True)

    if run_simulations:
        for graph_type in graph_utils.GraphType:
            # prepare args
            args = {}
            if graph_type == graph_utils.GraphType.ERDOS_RENYI:
                args = {
                    'edge_probability': p_ER,
                    'seed': seed
                }

            # create graph
            graph, weighted_adj = graph_utils.create_graph_with_metropolis_hastings_weights(NN, graph_type, args)

            fig, axs = plt.subplots(figsize=(7, 7), nrows=2, ncols=2)
            title = f"Graph Type = {graph_type}"
            fig.suptitle(title)
            fig.canvas.manager.set_window_title(title)

            plot_utils.show_graph_and_adj_matrix(fig, axs[0], graph, weighted_adj)
            
            # run gradient tracking with previously defined zz_init and cost_functions
            cost, grad, zz, ss = gradient_tracking(max_iter, NN, dd, zz_init, weighted_adj, cost_functions, alpha)

            plot_utils.show_cost_evolution(axs[1][0], cost, max_iter, semilogy=True, cost_opt=cost_opt)
            plot_utils.show_norm_of_total_gradient(axs[1][1], grad, max_iter, semilogy=True)
            plot_utils.show_and_wait(fig)
    

    # run set of simulations with different graphs (topology) with weights determined by Metropolis-Hasting.
    # for each simulation:
    # - comparison with the "centralized" version
    # - plots: (log scale)
    #    - evolution of the "cost"
    #    - "consensus error"
    #    - norm of the TOTAL gradient



    