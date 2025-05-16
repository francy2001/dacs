import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

seed = 42
np.random.seed(seed)

max_iter = 500

def quadratic_cost_fn(zz, QQ, rr):
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(NN, d, zz, ss, weighted_adj, cost_functions):
    cost = np.zeros((max_iter))

    for k in range(max_iter - 1):
        for i in range(NN):
            zz[k+1, i] = zz[k].T @ weighted_adj[i].T - alpha * ss[k, i]
            # TODO: maybe reversed? zz[k+1, i] = (weighted_adj[i] @ zz[k]).T - alpha * ss[k, i]

            consensus = ss[k].T @ weighted_adj[i].T
            
            cost_k_i, grad_k_i = cost_functions[i](zz[k, i])
            _, grad_k_plus_1_i = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_k_plus_1_i - grad_k_i
            ss[k+1, i] = consensus + local_innovation

            cost[k] += cost_k_i

    return cost, zz, ss
    
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

def create_graph_erdos_renyi_metropolis_hastings_weights(NN, p_er):
    global seed

    ONES = np.ones((NN, NN))
    while True:
        G = nx.erdos_renyi_graph(NN, p_er, seed)
        Adj = nx.adjacency_matrix(G).toarray()
        is_strongly_connected = np.all(np.linalg.matrix_power(Adj + np.eye(NN), NN) > 0)
        
        if is_strongly_connected:
            break

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


if __name__ == "__main__":
    NN = 10     # number of agents
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

    # init s
    for i in range(NN):
        _, grad = quadratic_cost_fn(z[0, i], QQ_list[i], rr_list[i])
        s[0, i] = grad


    graph, weighted_adj = create_graph_erdos_renyi_metropolis_hastings_weights(NN, p_ER)
    show_graph_and_adj_matrix(graph, weighted_adj)

    cost_functions = []
    for i in range(NN):
        cost_functions.append(lambda zz, i=i: quadratic_cost_fn(zz, QQ_list[i], rr_list[i]))

    cost, zz, ss = gradient_tracking(NN, d, z, s, weighted_adj, cost_functions)

    # graph config: [1]
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)

    ax = axes[0]
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.set_title("Cost error")
    ax.plot(np.arange(max_iter - 1), cost[:-1])
    ax.plot(np.arange(max_iter - 1), cost_opt * np.ones((max_iter - 1)), "r--")

    ax = axes[1]
    z_avg = np.mean(z, axis=1)
    ax.set_title("Consensus error")
    for i in range(NN):
        # distance for each agent from the average of that iteration
        ax.plot(np.arange(max_iter), z[:, i] - z_avg)
        
    plt.show(block=False)

    # graph config: [2]
    # we can appreciate the LINEAR rate of convergence!
    fig, axes = plt.subplots(figsize=(8, 6), nrows=1, ncols=2)

    ax = axes[0]
    ax.set_title("Cost error")
    # optimal cost error - one line! we are minimizing the sum not each l_i
    ax.semilogy(np.arange(max_iter - 1), np.abs(cost[:-1] - cost_opt))

    ax = axes[1]
    z_avg = np.mean(z, axis=1)
    ax.set_title("Consensus error")
    for i in range(NN):
        ax.semilogy(np.arange(1, max_iter), np.abs(z[1:, i] - z_avg[1:])) # NOTE: skipping k=0
        
    plt.show()

    # run set of simulations with different graphs (topology) with weights determined by Metropolis-Hasting.
    # for each simulation:
    # - comparison with the "centralized" version
    # - plots: (log scale)
    #    - evolution of the "cost"
    #    - "consensus error"
    #    - norm of the TOTAL gradient



    