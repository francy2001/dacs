import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils, animation

seed = 38
np.random.seed(seed)

# Global Parameters
max_iter = 450

def cost_fn(zz, rr, barycenter, gamma_1, gamma_2):
    """
    Compute the cost function for the aggregative tracking problem.
    
    Parameters
    ----------
    zz : np.ndarray
        The positions of the i-th agents at the current iteration.
    rr : np.ndarray
        The target positions of the i-th agents.
    barycenter : np.ndarray
        The barycenter of the agents' positions.
    gamma_1 : float
        Weight for the first term of the cost function (distance to target).
    gamma_2 : float
        Weight for the second term of the cost function (distance to barycenter).        

    Returns
    -------
    cost : float
        The computed cost value.
    """
    target_norm = np.linalg.norm(zz-rr)
    barycenter_norm = np.linalg.norm(zz - barycenter)

    # print(f"Target_norm: {target_norm}, shape: {target_norm.shape}")
    # print(f"barycenter_norm: {barycenter_norm}, shape: {barycenter_norm.shape}")

    cost = gamma_1 * target_norm**2 + gamma_2 * barycenter_norm**2

    return cost


def aggregative_variable(zz): # barycenter of the agents' positions
    """
    Compute the aggregative variable (barycenter) of the agents' positions.
    
    Parameters
    ----------
    zz : np.ndarray
        The positions of the agents.
    
    Returns
    -------
    barycenter : np.ndarray
        The barycenter of the agents' positions.
    """
    barycenter = np.mean(zz, axis=0)  # Compute the mean of the agents' positions
    return barycenter

def gradient_computation(zz, rr, barycenter, gamma_2, N, type, gamma_1=None):
    if type == 'first':
        # derivate of the cost function with respet to zz
        grad = 2 * gamma_1 * (zz - rr) + 2 * gamma_2 * (1 - 1/N) * (zz - barycenter)
    elif type == 'second':
        # derivate of the cost function with respect to sigma(z)
        grad = -2 * gamma_2 * (zz - barycenter)
    else:
        raise ValueError("Invalid type. Use 'first' or 'second'.")
    return grad

def centralized_aggregative_method(alpha: float, z_init: np.ndarray, target_pos: np.ndarray, dim: tuple, cost_functions, gamma_1, gamma_2):
    """
    Centralized Aggregative algorithm.

    Parameters
    ----------
    alpha : float
        Stepsize
    z_init : numpy.ndarray
        A (N, d) array that contains the initial state of the agents.
    target_pos : numpy.ndarray
        A (N, d) array that contains the personal known fixed targets for each agent i.
    dim : tuple of int
        A tuple that contains (max_iter, N, d)
    cost_functions : list of callable
        A list of cost functions for each agent, where each function takes two arguments: the current position of the agent and the barycenter.
    gamma_1 : numpy.ndarray
        A (N,) array that contains the weights for the first term of the cost function for each agent.
    gamma_2 : numpy.ndarray
        A (N,) array that contains the weights for the second term of the cost function for each agent. 
        
    Returns
    -------
    numpy.ndarray
        Return zz of shape=dim containing the evolution.

    Raises
    ------
    ValueError
        If the shapes of `z_init` and `target_pos` do not match the specified ...
    """
    
    (max_iter, N, d) = dim
    zz = np.zeros((max_iter, N, d))  # state: positions of the agents
    cost = np.zeros(max_iter)
    grad = np.zeros((max_iter, d))
    
    zz[0] = z_init

    def centralized_cost_fn(zz, barycenter):
        """
        Parameters
        ----------
            zz: np.ndarray
                shape=(N,d) the agents' states
            barycenter: np.ndarray
                shape=(d,), the mean of the positions of all team members
        """
        total_cost = 0
        for i in range(N):
            c = cost_functions[i](zz[i], barycenter)
            total_cost += c
        return total_cost
    
    for k in range(max_iter - 1):
        barycenter = aggregative_variable(zz[k])    # Compute the barycenter of the agents' positions
        cost[k] = centralized_cost_fn(zz[k], barycenter)
        # print(f"Centralized cost at iteration {k}: {cost[k]}")
        for i in range(N):
            gradient_sum = np.zeros(d) 
            for j in range(N):
                gradient_sum += gradient_computation(zz[k,j], target_pos[j], barycenter, gamma_2=gamma_2[j], N=N, type='second')
            nabla_1 = gradient_computation(zz[k,i], target_pos[i], barycenter, gamma_1=gamma_1[i], gamma_2=gamma_2[i], N=N, type='first')
            grad_i = nabla_1 + np.eye(d) @ gradient_sum
            zz[k+1, i] = zz[k,i] - alpha * grad_i
            
            grad[k] += grad_i

    return cost, grad, zz

def aggregative_tracking(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, adj, safety):

    """ 
    Distributed Aggregative Tracking algorithm.

    Parameters
    ----------
    alpha : float
        Stepsize
    z_init : numpy.ndarray
        A (N, d) array that contains the initial state of the agents.
    target_pos : numpy.ndarray
        A (N, d) array that contains the personal known fixed targets for each agent i.
    dim : tuple of int
        A tuple that contains (max_iter, N, d)
    cost_functions : list of callable
        A list of cost functions for each agent, where each function takes two arguments: the current position of the agent and the barycenter.
    gamma_1 : numpy.ndarray
        A (N,) array that contains the weights for the first term of the cost function for each agent.
    gamma_2 : numpy.ndarray
        A (N,) array that contains the weights for the second term of the cost function for each agent.
    adj : numpy.ndarray
        A (N, N) adjacency matrix representing the communication graph between the agents.
    safety : bool
        If True, the safety controller is applied to ensure that the agents' positions remain within a safe distance from each other.

    Returns
    -------
    tuple
        A tuple containing:
        - total_cost: numpy.ndarray
            An array of shape (max_iter,) containing the total cost at each iteration.
        - total_grad: numpy.ndarray
            An array of shape (max_iter, d) containing the total gradient at each iteration.
        - zz: numpy.ndarray
            An array of shape (max_iter, N, d) containing the positions of the agents at each iteration.
        - ss: numpy.ndarray
            An array of shape (max_iter, N, d) containing the proxy of \sigma(x^k) at each iteration.
        - vv: numpy.ndarray
            An array of shape (max_iter, N, d) containing the proxy of \frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_J(z_j^k, \sigma(z^k)) at each iteration.   
    """

    from safety_controller import safety_controller, neighborhood_distances # safety_controller is imported to use the safety controller inside the function for circular import issues

    # 3 states
    (max_iter, N, d) = dim
    zz = np.zeros((max_iter, N, d))  # positions of the agents
    ss = np.zeros((max_iter, N, d))  # proxy of \sigma(x^k)
    vv = np.zeros((max_iter, N, d))  # proxy of \frac{1}{N}\sum_{j=1}^{N}\nabla_2\ell_J(z_j^k, \sigma(z^k))
    xx = np.zeros((max_iter, N, d))  # single integrator of the agents' positions

    total_cost = np.zeros(max_iter)
    total_grad = np.zeros((max_iter, d))

    # Initialization
    zz[0] = z_init 
    ss[0] = z_init # \phi_{i}(z) is the identity function
    xx[0] = z_init 
    for i in range(N):
        vv[0, i] = gradient_computation(zz[0,i], target_pos[i], ss[0, i], gamma_2=gamma_2[i], N=N, type='second')

    for k in range(max_iter-1):
        for i in range(N):
            # NOTE: usage of ss[k,i] instead of barycenter
            cost = cost_functions[i](zz[k,i], ss[k,i])
            
            # [ zz update ]
            nabla_1 = gradient_computation(zz[k,i], target_pos[i], ss[k,i], gamma_1=gamma_1[i], gamma_2=gamma_2[i], N=N, type='first')
            grad = nabla_1 + np.eye(d) @ vv[k,i]

            total_cost[k] += cost
            total_grad[k] += grad
    
            if safety:      # Safety controller

                delta_t=0.1
                z_temp = zz[k, i] - alpha * grad
                u_ref = z_temp
                neighbors_distances = neighborhood_distances(xx[k], i)  # Get distances of neighbors    
                u_safe = safety_controller(u_ref, neighbors_distances)  # Apply the safety controller to the reference control input
                
                if u_safe is None:
                    zz[k+1, i] = xx[k, i]  # If the safe control input is None, stop in the current position
                else:
                    xx[k+1, i] = xx[k, i] + delta_t * (u_safe - xx[k,i])  # Update the position with the safe control input
                    zz[k+1, i] = xx[k+1, i]

            else:
                zz[k+1, i] = zz[k, i] - alpha * grad
                
            # [ ss update ]
            ss_consensus = ss[k].T @ adj[i].T
            ss_local_innovation = zz[k+1, i] - zz[k, i]
            ss[k+1, i] = ss_consensus + ss_local_innovation

            # [ vv update ]
            vv_consensus = vv[k].T @ adj[i].T
            nabla2_k = gradient_computation(zz[k,i], target_pos[i], ss[k,i], gamma_2=gamma_2[i], N=N, type='second')
            nabla2_k_plus_1 = gradient_computation(zz[k+1,i], target_pos[i], ss[k+1,i], gamma_2=gamma_2[i], N=N, type='second')
            vv_local_innovation = nabla2_k_plus_1 - nabla2_k
            vv[k+1, i] = vv_consensus + vv_local_innovation

    return total_cost, total_grad, zz, ss, vv

def __sim_aggregative_tracking(title, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type, args={}, vip_idx=None):
    # [ graph creation ]
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_type, args)
    
    fig, axs = plt.subplots(figsize=(15,10), nrows=2, ncols=2)
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title)

    plot_utils.show_graph_and_adj_matrix(fig, axs[0], graph, adj)
    
    # [ centralized gradient tracking ]
    cost_centr, grad_centr, zz_centr = centralized_aggregative_method(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2)

    # [ distriuted gradient tracking ]
    res = aggregative_tracking(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, adj, safety=False)
    (total_cost_distr, total_grad_distr, zz_distr, ss_distr, vv_distr) = res

    # [ plots ]
    plot_utils.show_cost_evolution(axs[1][0], cost_centr, max_iter, semilogy=True, label="Centralized")
    plot_utils.show_cost_evolution(axs[1][0], total_cost_distr, max_iter, semilogy=True, label="Distributed")

    plot_utils.show_norm_of_total_gradient(axs[1][1], grad_centr, max_iter, semilogy=True, label="Centralized")
    plot_utils.show_norm_of_total_gradient(axs[1][1], total_grad_distr, max_iter, semilogy=True, label="Distributed")
    
    plot_utils.show_and_wait(fig)

    # [ show centralized ]
    # fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    # animation.animation(ax, zz_centr, target_pos, vip_idx=vip_idx, title=f"{title} \n Centralized")
    
    # [ show distributed ]
    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    animation.animation(ax, zz_distr, target_pos, adj=adj, vip_idx=vip_idx, title=f"{title} \n Distributed")

def __sim_network_graphs(title, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2):
    """ Run the Distributed Aggregative Tracking algorithms with different network graphs.
    """
    for graph_type in graph_utils.GraphType:
            args = {}
            if graph_type == graph_utils.GraphType.ERDOS_RENYI:
                args = {
                    'edge_probability': 0.65,
                    'seed': seed
                }            
            __sim_aggregative_tracking(title, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type, args)

if __name__ == "__main__":
    # setup 
    N = 4  # number of agents
    d = 2  # dimension of the state space

    # generate N target positions
    target_pos = np.random.uniform(low=-10, high=10, size=(N, d))
    print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))
    
    # generate initial positions for the agents
    z_init = np.random.uniform(low=-10, high=10, size=(N, d))
    print("Initial Positions: {}\tShape: {}".format(z_init, z_init.shape))

    args = {'edge_probability': 0.65, 'seed': seed}
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)

    # define ell_i
    gamma_1 = np.ones(N)    # equally distributed weights
    gamma_2 = np.ones(N)    # equally distributed weights
    cost_functions = []
    for i in range(N):
        cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))

    alpha = 0.05 # step size
    dim = (max_iter, N, d)

    # -----------------------
    # |     CENTRALIZED     |
    # -----------------------
    cost_centr, grad_centr, zz_centr = centralized_aggregative_method(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2)

    # -----------------------
    # |     DISTRIBUTED     |
    # -----------------------
    res = aggregative_tracking(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, adj, safety=False)
    (total_cost_distr, total_grad_distr, zz_distr, ss_distr, vv_distr) = res

    # run simulations
    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

    ax = axes[0]
    plot_utils.show_cost_evolution(ax, cost_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_cost_evolution(ax, total_cost_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    ax = axes[1]
    plot_utils.show_norm_of_total_gradient(ax, grad_centr, max_iter, semilogy=True, label="Centralized Aggregative Tracking")
    plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    # ax = axes[2]
    # plot_utils.show_cost_evolution(ax, ss_distr[:,:,0], max_iter,semilogy=False, label="Distributed Aggregative Trackgin - vv")
    # plot_utils.show_consensus_error(ax, N, zz_centr, semilogy=False, label="Consensus Error Centralized")
    # plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")

    plot_utils.show_and_wait(fig)

    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    animation.animation(ax, zz_centr, target_pos, title="Centralized")

    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    animation.animation(ax, zz_distr, target_pos, adj=adj, title="Distributed")

    # fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    # animation.animation(ax, vv_distr, target_pos, adj, title="Distributed")


    # -------------------------
    # |      SIMULATIONS      |
    # -------------------------
    simulation = True

    if simulation:
        print("Simulations...")

        args = {'edge_probability': 0.65, 'seed': seed}

        # [ different graphs ]
        def sim_network_graphs():
            gamma_1 = np.ones(N)    # equally distributed weights
            gamma_2 = np.ones(N)    # equally distributed weights
            cost_functions = []
            for i in range(N):
                cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))
            __sim_network_graphs("Different graphs", z_init, target_pos, dim, cost_functions, gamma_1, gamma_2)

        # [ vip agent ]
        def sim_vip_agent():
            vip_idx = 0
            gamma_1 = np.ones(N)
            gamma_2 = np.ones(N)
            # TODO: NOTE: with gamma1=0 the centralized diverges!!! it remains a small drift, so a steady-state error (?)
            gamma_1[vip_idx] = 0     # ignore the correspondent target
            gamma_2[vip_idx] = 2     # stay close to the barycenter!
            cost_functions = []
            for i in range(N):
                cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))
            __sim_aggregative_tracking("VIP agent", z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type=graph_utils.GraphType.ERDOS_RENYI,args=args, vip_idx=vip_idx)

        # [ closer to targets ]
        def sim_closer_to_targets():
            gamma_1 = np.ones(N) * 4
            gamma_2 = np.ones(N)
            cost_functions = []
            for i in range(N):
                cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))
            __sim_aggregative_tracking("Closer to targets", z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type=graph_utils.GraphType.ERDOS_RENYI, args=args)

        # [ more team cohesion ]
        def sim_more_cohesion():
            gamma_1 = np.ones(N)
            gamma_2 = np.ones(N) * 2
            cost_functions = []
            for i in range(N):
                cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))
            __sim_aggregative_tracking("Team Cohesion", z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type=graph_utils.GraphType.CYCLE)
        
        
        sim_network_graphs()
        sim_vip_agent()
        sim_closer_to_targets()
        sim_more_cohesion()
        
