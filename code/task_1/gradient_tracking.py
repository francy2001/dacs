import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from enum import Enum

seed = 48
np.random.seed(seed)

max_iter = 500
simulation = False

class GraphType(Enum):
    ERDOS_RENYI = 1
    CYCLE = 2
    PATH = 3
    STAR = 4
    COMPLETE = 5

def quadratic_cost_fn(zz, QQ, rr):
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(NN, Nt, d, zz, ss, weighted_adj, cost_functions, alpha):
    max_iter = zz.shape[0]
    cost = np.zeros((max_iter))
    grad = np.zeros((max_iter, Nt, d))
    grad = grad.squeeze()

    print(f"zz[0]: {zz[0]}")

    for k in range(max_iter - 1):
        print(f"k: {k}")
        for i in range(NN):
            # print(f"zz[k].shape: {zz[k].shape}")
            print(f"zz[k][0][0]: {zz[k][0][0]}")
            # print(f"np.transpose(zz[k]).shape: {np.moveaxis(zz[k], 0, -1).shape}")
            # print(f"ss[k,i].shape: {ss[k,i].shape}")
            # print(f"weighted_adj[i].T.shape: {weighted_adj[i].T.shape}")

            zz_k_T = np.moveaxis(zz[k], 0, -1)
            # print(f"zz_k_T : {zz_k_T}")
            # print(f"zz[k].T : {zz[k].T}")
            # print(f"np.transpose(zz[k], axes=[1,2,0]): {np.transpose(zz[k], axes=[1,2,0])}")

            # zz[k+1, i] = zz[k].T @ weighted_adj[i].T - alpha * ss[k, i]
            mul = zz_k_T @ weighted_adj[i].T
            mul = mul.squeeze()
            zz[k+1, i] = mul - alpha * ss[k, i]

            # TODO: maybe reversed? zz[k+1, i] = (weighted_adj[i] @ zz[k]).T - alpha * ss[k, i]

            ss_k_T = np.moveaxis(ss[k], 0, -1)
            print(f"ss_k_T : {ss_k_T}")
            print(f"ss[k].shape : {ss[k].shape}")
            # consensus = ss[k].T @ weighted_adj[i].T
            consensus = ss_k_T @ weighted_adj[i].T
            consensus = consensus.squeeze()
            
            cost_k_i, grad_k_i = cost_functions[i](zz[k, i])
            _, grad_k_plus_1_i = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_k_plus_1_i - grad_k_i
            ss[k+1, i] = consensus + local_innovation

            cost[k] += cost_k_i
            grad[k] += grad_k_i

        # if k == 5:
        #     break

    return cost, grad, zz, ss
    
def metropolis_hastings_weights(graph):
    N = graph.number_of_nodes()
    A = np.zeros(shape=(N, N), dtype='float64')
    for i in range(A.shape[0]):
        N_i = list(graph.neighbors(i))
        d_i = len(N_i)
        for j in range(A.shape[0]):
            N_j = list(graph.neighbors(j))
            d_j = len(N_j)
            if i == j: 
                sum = 0
                for h in N_i:
                    sum += A[i, h]
                A[i,j] = 1 - sum
            elif graph.has_edge(i,j):
                A[i,j] = 1 / (1 + max(d_i, d_j))

    # Normalize
    max_iterations = 1000
    tolerance = 10e-9
    for _ in range(max_iterations):
        A = A / np.sum(np.abs(A), axis=1, keepdims=True)
        A = A / np.sum(np.abs(A), axis=0, keepdims=True)
        A = np.abs(A)

        # Check for convergence
        if np.all(np.sum(A, axis=1) - 1 < tolerance) and np.all(np.sum(A, axis=0) - 1 < tolerance):
            break

    return A

def create_graph_with_metropolis_hastings_weights(NN, graph_type, args={}):
    global seed

    ONES = np.ones((NN, NN))
    
    if graph_type == GraphType.ERDOS_RENYI:
        p_er = args['edge_probability']
        while True:
            G = nx.erdos_renyi_graph(NN, p_er, seed)
            Adj = nx.adjacency_matrix(G).toarray()
            is_strongly_connected = np.all(np.linalg.matrix_power(Adj + np.eye(NN), NN) > 0)
            
            if is_strongly_connected:
                break

    elif graph_type == GraphType.CYCLE:
        G = nx.cycle_graph(NN)

    elif graph_type == GraphType.PATH:
        G = nx.path_graph(NN)
    
    elif graph_type == GraphType.STAR:
        G = nx.star_graph(NN - 1)

    elif graph_type == GraphType.COMPLETE:
        G = nx.complete_graph(NN)

    A = metropolis_hastings_weights(G)

    return G, A

def create_graph_birkhoff_von_neumann(NN, num_vertices):
    I_n = np.eye(NN)
    
    # key=hash; value=np.ndarray
    vertices = {}
    
    # Number of ways to choose k items from n items without repetition and with order.
    max = math.perm(NN, NN)
    print(f"Number of possibles vertices: {math.perm(NN, NN)}") # math.factorial(NN)
    assert num_vertices <= max, "Vertices won't be unique"

    # Ensure the presence of the Identity Matrix, self-loops
    vertices[hash(I_n.tobytes())] = I_n

    while(len(vertices) < num_vertices):
        vertix = np.random.permutation(I_n)
        vertix_hash = hash(vertix.tobytes())
        
        if vertix_hash not in vertices:
            vertices[vertix_hash] = vertix
    
    # k=class_weights, the len(k) is the number of extracted numbers
    class_weights = np.ones((num_vertices)) # equally distributed classes

    convex_coefficients = np.random.dirichlet(alpha=class_weights, size=1).squeeze()
    # print(f"convex_coefficients: {convex_coefficients}")

    doubly_stochastic_matrix = np.zeros((NN, NN))
    vertices = list(vertices.values())
    # print(f"vertices: {vertices}")
    
    for i in range(num_vertices):
        doubly_stochastic_matrix += vertices[i] * convex_coefficients[i]

    # Ensure symmetry i.e. undirected graph
    doubly_stochastic_matrix = (doubly_stochastic_matrix + doubly_stochastic_matrix.T) / 2

    # print(doubly_stochastic_matrix)
    # print(np.sum(doubly_stochastic_matrix, axis=0))
    # print(np.sum(doubly_stochastic_matrix, axis=1))

    paths_up_to_N = np.linalg.matrix_power(doubly_stochastic_matrix + np.eye(NN), NN)

    # if is full, then in strongly connected
    if np.all(paths_up_to_N > 0):
        print("The graph is strongly connected")
    else: 
        print("The graph is NOT strongly connected!")

    graph = nx.from_numpy_array(doubly_stochastic_matrix)

    return graph, doubly_stochastic_matrix

def show_graph_and_adj_matrix(graph, adj_matrix=None):

    fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
    nx.draw(graph, with_labels=True, ax=axs[0])

    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph).toarray()
    
    cax = axs[1].matshow(adj_matrix, cmap='plasma')# , vmin=0, vmax=1)
    fig.colorbar(cax)

    plt.show(block=False)

def show_simulations_plots(cost, cost_opt, grad):

    fig, axs = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)

    ax = axs[0]
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.set_title("Cost")
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1]))
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost_opt * np.ones((max_iter - 1))), "r--")

    ax = axs[1]
    ax.set_title("Norm of the total gradient - log scale")
    ax.semilogy(np.arange(max_iter - 1), np.linalg.norm(grad[:-1], axis=1)**2)

    plt.show(block=False)

if __name__ == "__main__":
    # TODO: parametri a linea di comando
    NN = 10     # number of agents
    Nt = 1
    d = 2       # dimension of the state
    p_ER = 0.65 

    QQ_list = []
    rr_list = []

    for i in range(NN):
        Q = np.diag(np.random.uniform(size=d)) # TODO: use SVD, syntetize rotations
        r = np.random.uniform(size=d)

        QQ_list.append(Q)
        rr_list.append(r)

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

    # two states
    z = np.zeros((max_iter, NN, d)) # indeces: [time, who, state-component]
    s = np.zeros((max_iter, NN, d))

    # init z
    z_init = np.random.normal(size=(NN, d))
    z[0] = z_init
    # print(f"z_init: {z_init}")

    # init s
    for i in range(NN):
        _, grad = quadratic_cost_fn(z[0, i], QQ_list[i], rr_list[i])
        s[0, i] = grad

    # general case of a graph with birkhoff-von-neumann weights
    graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.COMPLETE)
    show_graph_and_adj_matrix(graph, weighted_adj)

    cost_functions = []
    for i in range(NN):
        cost_functions.append(lambda zz, i=i: quadratic_cost_fn(zz, QQ_list[i], rr_list[i]))

    cost, grad, zz, ss = gradient_tracking(NN, Nt, d, z, s, weighted_adj, cost_functions, alpha)
    # print(f"l[0:10]: {cost[0:10]}")
    # print(f"cost[230:280] - cost_opt:{cost[230:280] - cost_opt}")

    print(f"cost.shape: {cost.shape}")

    # graph config: [1]
    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=4)

    fig.suptitle("Plot - Linear scale")
    fig.canvas.manager.set_window_title("Plot - Linear scale")

    ax = axes[0]
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.set_title("Cost")
    ax.plot(np.arange(max_iter - 1), cost[:-1])
    ax.plot(np.arange(max_iter - 1), cost_opt * np.ones((max_iter - 1)), "r--")

    ax = axes[1]
    z_avg = np.mean(z, axis=1)
    ax.set_title("Consensus error")
    for i in range(NN):
        # distance for each agent from the average of that iteration
        ax.plot(np.arange(max_iter), z[:, i] - z_avg)

    ax = axes[2]
    ax.set_title("Cost error")
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.plot(np.arange(max_iter - 1), cost[:-1] - cost_opt)

    ax  = axes[3]
    ax.set_title("Norm of the total gradient")
    ax.plot(np.arange(max_iter - 1), np.linalg.norm(grad[:-1], axis=1)**2)
        
    plt.show(block=False)

    fig, axes = plt.subplots(figsize=(15, 10), nrows=1, ncols=4)

    fig.suptitle("Plot - Logarithmic scale")
    fig.canvas.manager.set_window_title("Plot - Logarithmic scale")

    ax = axes[0]
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.set_title("Cost")
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1]))
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost_opt * np.ones((max_iter - 1))), "r--")

    ax = axes[1]
    z_avg = np.mean(z, axis=1)
    ax.set_title("Consensus error")
    for i in range(NN):
        # distance for each agent from the average of that iteration
        ax.semilogy(np.arange(max_iter), np.abs(z[:, i] - z_avg))

    ax = axes[2]
    ax.set_title("Cost error")
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1] - cost_opt))

    ax = axes[3]
    ax.set_title("Norm of the total gradient - log scale")
    ax.semilogy(np.arange(max_iter - 1), np.linalg.norm(grad[:-1], axis=1)**2)


    if simulation:

        for en in GraphType:
            if en == GraphType.ERDOS_RENYI:
                graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.ERDOS_RENYI, {'edge_probability': p_ER})
            elif en == GraphType.CYCLE:
                graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.CYCLE)
            elif en == GraphType.PATH:
                graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.PATH)
            elif en == GraphType.STAR:
                graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.STAR)
            elif en == GraphType.COMPLETE:
                graph, weighted_adj = create_graph_with_metropolis_hastings_weights(NN, GraphType.COMPLETE)

            show_graph_and_adj_matrix(graph, weighted_adj)
            cost_functions = []
            for i in range(NN):
                cost_functions.append(lambda zz, i=i: quadratic_cost_fn(zz, QQ_list[i], rr_list[i]))

            cost, grad, zz, ss = gradient_tracking(NN, d, z, s, weighted_adj, cost_functions, alpha)

            show_simulations_plots(cost, cost_opt, grad)

    plt.show()

    # run set of simulations with different graphs (topology) with weights determined by Metropolis-Hasting.
    # for each simulation:
    # - comparison with the "centralized" version
    # - plots: (log scale)
    #    - evolution of the "cost"
    #    - "consensus error"
    #    - norm of the TOTAL gradient



    