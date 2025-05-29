import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils

seed = 42
np.random.seed(seed)

# implement of the aggregative tracking algorithm
max_iter = 1000

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

# def aggregative_tracking()


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

# def aggregative_tracking(...):
#     break

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
    print(f"grad shape: {grad.shape}")
    return grad


if __name__ == "__main__":
    # setup 
    N = 3  # number of agents
    d = 2  # dimension of the state space

    # three states
    z = np.zeros((max_iter, N, d))  # positions of the agents
    s = np.zeros((max_iter, N, d))  
    v = np.zeros((max_iter, N, d))  

    # generate N target positions
    target_pos = np.random.rand(N, d) * 10
    print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))
    # generate initial positions for the agents
    z_init = np.random.normal(size=(N, d)) * 10
    z[0] = z_init
    print("Initial Positions: {}\tShape: {}".format(z_init, z_init.shape))

    # fig = plt.figure()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # ax = fig.gca()
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(-10, 10)
    # ax.plot(z_init[:,0], z_init[:,1], 'ro', label='Robot Positions')
    # ax.plot(target_pos[:,0], target_pos[:,1], 'bx', label='Target Positions')
    # plt.grid()
    # plot_utils.show_and_wait(fig)

    # define ell_i
    gamma_1 = np.ones(N)  # equally distributed weights
    gamma_2 = np.ones(N)  # equally distributed weights
    cost_functions = []
   
    for i in range(N):
        cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))

    def centralized_cost_fn(zz):
        total_cost = 0
        barycenter = aggregative_variable(zz)  # Compute the barycenter of the agents' positions
        for i in range(N):
            c = cost_functions[i](zz[i], barycenter)
            total_cost += c
        return total_cost
    
    # -------------------------
    # |     CENTRALIZED    |
    # -------------------------
    alpha = 0.05 # step size
    
    z_centralized = np.zeros((max_iter, N, d)) 
    z_init_centr = z_init
    z_centralized[0] = z_init_centr
    print(f"Centralized initial position: {z_init_centr}")
    print(f"Centralized zz: {z_centralized}")

    cost_centr = np.zeros(max_iter)  

    for k in range(max_iter -1):
        barycenter = aggregative_variable(z_centralized[k])  # Compute the barycenter of the agents' positions
        cost_centr[k] = centralized_cost_fn(z_centralized[k])
        print(f"Centralized cost at iteration {k}: {cost_centr[k]}")
        for i in range(N):
            gradient_sum = np.zeros(d) 
            for j in range(N):
                gradient_sum += gradient_computation(z_centralized[k,j], target_pos[j], barycenter, gamma_1[j], gamma_2[j], N, type='second')
            nabla_1 = gradient_computation(z_centralized[k,i], target_pos[i], barycenter, gamma_1[i], gamma_2[i], N, type='first')
            z_centralized[k+1, i] = z_centralized[k,i] - alpha * ( nabla_1 + np.eye(d) @ gradient_sum )

    # define ell_i
    # gamma = np.random.uniform(0.1, 1.0, size=N)  # weights for the cost functions
    # gamma_2 = np.random.uniform(0.1, 1.0, size=N)  # random weights for the cost functions


    # run simulations

    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

    ax = axes[0]
    # optimal cost error - one line! we are minimizing the sum not each l_i
    plot_utils.show_cost_evolution(ax, cost_centr, max_iter, semilogy=True, label="Centralized")

    ax = axes[1]
    plot_utils.show_cost_evolution(ax, cost_centr, max_iter, semilogy=False, label="Centralized")

    plot_utils.show_and_wait(fig)

    fig2 = plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    ax = fig2.gca()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.plot(z_centralized[max_iter-1,:,0], z_centralized[max_iter-1,:,1], 'ro', label='Robot Positions')
    ax.plot(target_pos[:,0], target_pos[:,1], 'bx', label='Target Positions')
    plt.grid()

    fig = plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    ax = fig.gca()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.plot(z_init[:,0], z_init[:,1], 'ro', label='Robot Positions')
    ax.plot(target_pos[:,0], target_pos[:,1], 'bx', label='Target Positions')
    plt.grid()

    plt.show()



    # pass