import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils

seed = 38
np.random.seed(seed)

max_iter = 1000
run_simulations = False

def quadratic_cost_fn(zz, QQ, rr):
    """
    Quadratic cost function and its gradient.

    Parameters:
    ----------
        zz: np.ndarray
            The state vector for the agent.
        QQ: np.ndarray
            The quadratic term matrix.
        rr: np.ndarray
            The linear term vector.
    
    Returns:
    -------
        cost: float
            The value of the quadratic cost function.
            l(zz) = 0.5 * zz^T * QQ * zz + rr^T * zz
        grad: np.ndarray
            The gradient of the cost function with respect to zz.
            grad(l(zz)) = QQ * zz + rr
    """
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(max_iter, NN, dd: tuple, zz_init, weighted_adj, cost_functions, alpha):
    """
    Implementation of the Gradient Tracking Algorithm.
    
    Parameters:
    ----------
        max_iter: int
            The maximum number of iterations for the algorithm.
        NN: int
            The number of agents in the system.
        dd: tuple
            For multidimensional state, e.g. dd=(Nt, d), each agents has a state containing Nt positions in R^d, one for each target
        zz_init: np.ndarray
            Initial state for each agent, shape (NN, *dd).
        weighted_adj: np.ndarray
            The weighted adjacency matrix representing the communication graph between agents.
        cost_functions: list of callable
            List of cost functions for each agent, each function takes zz as input and returns (cost, grad).
        alpha: float
            Step size for the gradient descent update.
    
    Returns:
    -------
        cost: np.ndarray
            The cost at each iteration, shape (max_iter,).
        grad: np.ndarray
            The gradient at each iteration, shape (max_iter, *dd).
        zz: np.ndarray
            The state of each agent at each iteration, shape (max_iter, NN, *dd).
        ss: np.ndarray
            The tracker of the gradients at each iteration, shape (max_iter, NN, *dd).    
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
    
        for i in range(NN):

            # [ zz update ]
            # NOTE: we have a zz[k].shape: (N, *state_dim), we would like to multiply with a (N,) vector
            # Let's move the agent-axis in the last index such that we have: (*state_dim, N)@(N,1)-->(*state_dim,1)
            # If state_dim is only a tuple containing (d,) then it's equivalent to take the transpose.
            zz_k_T = np.moveaxis(zz[k], 0, -1)
            # print(f"zz_k_T.shape: {zz_k_T.shape}")
            zz[k+1, i] = zz_k_T @ weighted_adj[i].T - alpha * ss[k, i]

            # [ ss update ]
            # Move the agent-axis in the last index such that we have (Nt,d,N)@(N,1)-->(Nt,d,1)
            ss_k_T = np.moveaxis(ss[k], 0, -1)
            consensus = ss_k_T @ weighted_adj[i].T
            
            cost_k_i, grad_k_i = cost_functions[i](zz[k, i])
            _, grad_k_plus_1_i = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_k_plus_1_i - grad_k_i
            ss[k+1, i] = consensus + local_innovation

            # [ cost and gradient sums update ]
            cost[k] += cost_k_i
            grad[k] += grad_k_i 

        # grad[k] = sum of NN matrices with shape (Nt x d)

    return cost, grad, zz, ss


if __name__ == "__main__":
    NN = 10     # number of agents
    d = 2       # dimension of the state
    p_ER = 0.65 

    QQ_list = [] # list of positive definite matrices for the cost functions
    rr_list = [] # list of linear terms for the cost functions

    for i in range(NN):
        Q = np.diag(np.random.uniform(size=d)) # positive definite matrix
        r = np.random.uniform(size=d) # linear term

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



    