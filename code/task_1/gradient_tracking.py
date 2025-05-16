import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

seed = 42
np.random.seed(seed)


max_iter = 2000

def quadratic_cost_fn(zz, QQ, rr):
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(NN, d, zz, ss, weighted_adj, cost_functions):
    cost = np.zeros((max_iter))

    print(zz[0].T)
    print(zz[0].T.shape)
    print("-"*35)
    print(weighted_adj[0].T)
    print(weighted_adj[0].T.shape)
    print("-"*35)
    print(zz[0].T @ weighted_adj[0].T)
    print((zz[0].T @ weighted_adj[0].T).shape)


    for k in range(max_iter - 1):
        for i in range(NN):
            zz[k+1, i] = zz[k].T @ weighted_adj[i].T - alpha * ss[k, i]
            # zz[k+1, i] = (weighted_adj[i] @ zz[k]).T - alpha * ss[k, i]

            consensus = ss[k].T @ weighted_adj[i].T
            
            cost_k_i, grad_k_i = cost_functions[i](zz[k, i])
            _, grad_k_plus_1_i = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_k_plus_1_i - grad_k_i
            ss[k+1, i] = consensus + local_innovation

            cost[k] += cost_k_i

    return cost, zz, ss
    

def create_graph_iteratively(NN, p_er):
    ONES = np.ones((NN, NN))
    while 1:
        G = nx.erdos_renyi_graph(NN, p_er, seed)
        Adj = nx.adjacency_matrix(G).toarray()
        test = np.linalg.matrix_power(Adj + np.eye(NN), NN)

        # if is full, then in strongly connected
        if np.all(test > 0):
            break

    A = Adj + np.eye(NN)

    while any(abs(np.sum(A, axis=1) - 1) > 1e-10):
        A = A / (A @ ONES)
        A = A / (ONES.T @ A) # removing this part, I only get row-stochastcisty
        # they converge, but not at the minimum of the sum of the cost functions
        # but to a weighted avg of the sum

        A = np.abs(A)

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


    graph, weighted_adj = create_graph_birkhoff_von_neumann(NN, num_vertices=5)
    show_graph_and_adj_matrix(graph, weighted_adj)

    cost_functions = []
    for i in range(NN):
        cost_functions.append(lambda zz : quadratic_cost_fn(zz, QQ_list[i], rr_list[i]))

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



    