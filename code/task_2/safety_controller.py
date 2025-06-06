import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import networkx as nx
import sys
import os

from aggregative_tracking import aggregative_tracking, cost_fn 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import graph_utils, plot_utils, animation

seed = 38
np.random.seed(seed)

# Global Parameters
max_iter = 1000
delta = 0.5
world_dim = 10  # dimension of the world, i.e., the maximum value of the coordinates of the agents
gamma = 20+10*delta

def neighborhood_distances(zz, agent_index, radius=3*delta):
    """
    Centralized method to define the neighborhood of an agent based on its position zz and a given radius.
    All the agent whose distance is within the radius are considered neighbors.

    Parameters:
    zz : np.ndarray
        The positions of the agents.
    agent_index : int
        The index of the agent for which the neighborhood is defined.
    radius : float
        The radius within which agents are considered neighbors.
    
    Returns:
    distances : np.ndarray
        An array of distances from the agent at agent_index to its neighbors within the specified radius.
    """

    neighbors = np.delete(zz, agent_index, axis=0)  # Exclude the agent itself
    distances = []  # Initialize distances array
     
    print(f"Neighborhood of the agent {agent_index}")
    print(f"Positions of the neighbors: {neighbors}, shape: {neighbors.shape}")
    
    for j in range(neighbors.shape[0]):
        distance = (zz[agent_index] - neighbors[j])
        if np.linalg.norm(distance) <= radius:
            distances.append(distance) # Append the distance of the neighbor to the list
            # print(f"Agent {agent_index} is within radius {radius} of Agent {j}, distance: {np.linalg.norm(distance)}")

    return np.array(distances)

def safety_controller(u_ref, neighbors_dist):
    """
    Safety controller to ensure that the control input u_ref is safe with respect to the neighbors' positions distante from the agent neighbors_dist.
    
    Parameters:
    u_ref : np.ndarray
        The reference control input.
    neighbors_dist : np.ndarray
        The distances of the neighbors from the agent.   

    Returns:
    u_safe : np.ndarray
        The safe control input.
    """
    
    # No neighbors, return the reference control input
    if neighbors_dist.shape[0] == 0:
        print("No neighbors found, returning the reference control input.")
        return u_ref

    # Creation of the matrix A and b
    A = np.zeros((neighbors_dist.shape[0], u_ref.shape[0]))
    b = np.zeros(neighbors_dist.shape[0])

    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")
    # print(f"zz[i]: {zz[agent_index]}")

    for j in range(neighbors_dist.shape[0]):
        diff = neighbors_dist[j]  # Difference between the agent's position and the neighbor's position
        A[j] = -2 * diff.T
        b[j] = 0.5 * gamma *(np.linalg.norm(diff)**2 - delta**2)
    print(f"A: {A}, b: {b}")       

    # Check if the constraints are satisfied, if so, we don't need to solve the QP problem, we can just return u_ref
    print(f"Checking constraints for u_ref: {u_ref}")
    if np.all(A @ u_ref <= b):
        print(f"Constraints already satisfied for u_ref: {u_ref}")
        return u_ref

    # Solve the QP problem
    QQ = np.eye(u_ref.shape[0]) # Identity matrix for the quadratic term    
    qq = - u_ref                # Linear term

    minimum, cost = min_cvx_solver(QQ, qq, A, b)
    print(f"Minimum: {minimum}, Cost: {cost}")
    
    return minimum


def min_cvx_solver(QQ, qq, AA, bb):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/

    Obtain optimal solution for constrained QP

        min_{z} 1/2 z^T Q z + q^T z
        s.t.    Az - b <= 0

    """
    zz = cvx.Variable(qq.shape)

    # Quadratic cost function
    cost = 0.5* cvx.quad_form(zz,QQ) + qq.T @ zz     

    # Constraint Az <= b
    constraint = [AA@zz <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    solvers_to_try = ['OSQP', 'ECOS', 'CVXOPT', 'SCS']
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                print(f"Solver {solver} succeeded.")
                return zz.value, problem.value
        except Exception as e:
            continue
    return zz.value, problem.value


if __name__ == "__main__":
    # setup 
    N = 4  # number of agents
    d = 2  # dimension of the state space

    # generate N target positions
    target_pos = np.random.rand(N, d) * world_dim
    
    # generate initial positions for the agents
    z_init = np.random.rand(N, d) * world_dim
    
    # generate a communications graph    
    args = {'edge_probability': 0.65, 'seed': 38}
    graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
    
    print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))
    print("Initial Positions: {}\tShape: {}".format(z_init, z_init.shape))
    print("Adjacency Matrix: \n{}\nShape: {}".format(adj, adj.shape))

    # generate the cost function
    gamma_1 = np.ones(N)  # equally distributed weights
    gamma_2 = np.ones(N)  # equally distributed weights
    # gamma_1 = np.random.uniform(0.1, 1.0, size=N)     # random weights for the cost functions
    # gamma_2 = np.random.uniform(0.1, 1.0, size=N)     # random weights for the cost functions
    
    cost_functions = []
    for i in range(N):
        cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))

    alpha = 0.1 # step size
    dim = (max_iter, N, d)

    res = aggregative_tracking(alpha, z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, adj, safety=True)
    (total_cost_distr, total_grad_distr, zz_distr, ss_distr, vv_distr) = res

    # run simulations
    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)

    ax = axes[0]
    plot_utils.show_cost_evolution(ax, total_cost_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")
    ax = axes[1]
    plot_utils.show_norm_of_total_gradient(ax, total_grad_distr, max_iter, semilogy=True, label="Distributed Aggregative Tracking")
    plot_utils.show_and_wait(fig)

    fig, ax = plt.subplots(figsize=(15, 10), nrows=1, ncols=1)
    animation.animation(ax, zz_distr, target_pos, adj, safety_distance=delta)