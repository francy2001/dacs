import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

max_iter = 200

def quadratic_cost_fn(zz, QQ, rr):
    cost = 0.5 * zz.T @ QQ @ zz + rr.T @ zz
    grad = QQ @ zz + rr
    return cost, grad

def gradient_tracking(NN, d, zz, ss, weighted_adj, cost_functions):
    cost = np.zeros((max_iter))
    I_d = np.eye(d)
    weighted_adj_ext = np.kron(weighted_adj, I_d)

    for k in range(max_iter - 1):
        for i in range(NN):
            zz[k+1, i] = weighted_adj_ext[i] @ zz[0].reshape(-1).T - alpha * ss[k, i]
            consensus = weighted_adj_ext[i] @ ss[k].reshape(-1).T
            
            cost_ell_i_old, grad_ell_i_old = cost_functions[i](zz[k, i])
            _, grad_ell_i_new = cost_functions[i](zz[k+1, i])
            
            local_innovation = grad_ell_i_new - grad_ell_i_old
            ss[k+1, i] = consensus + local_innovation

            cost[k] += cost_ell_i_old

    return cost, zz, ss
    

def create_graph_iteratively(NN, p_er):
    ONES = np.ones((NN, NN))
    while 1:
        G = nx.erdos_renyi_graph(NN, p_er)
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

def show_graph_and_adj_matrix(graph, adj_matrix=None):
    fig, axs = plt.subplots(figsize=(6,3), nrows=1, ncols=2)
    nx.draw(graph, with_labels=True, ax=axs[0])

    if adj_matrix is None:
        adj_matrix = nx.adjacency_matrix(graph).toarray()
    
    cax = axs[1].matshow(adj_matrix, cmap='plasma')# , vmin=0, vmax=1)
    fig.colorbar(cax)
    plt.show()


if __name__ == "__main__":
    NN = 5     # number of agents
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
    alpha = 0.01

    # two states
    z = np.zeros((max_iter, NN, d)) # indeces: [time, who, state-component]
    s = np.zeros((max_iter, NN, d))

    # init z
    z_init = np.random.normal(size=(NN, d))
    z[0] = z_init

    # init s
    for i in range(NN):
        _, grad = quadratic_cost_fn(z[0,i], QQ_list[i], rr_list[i])
        s[0, i] = grad


    graph, weighted_adj = create_graph_iteratively(NN, p_er=p_ER)
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
        
    plt.show()

    # run set of simulations with different graphs (topology) with weights determined by Metropolis-Hasting.
    # for each simulation:
    # - comparison with the "centralized" version
    # - plots: (log scale)
    #    - evolution of the "cost"
    #    - "consensus error"
    #    - norm of the TOTAL gradient



    