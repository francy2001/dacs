from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np
from utils import graph_utils

seed = 38
np.random.seed(seed)

N = 3
d = 2
alpha = 0.05

max_iter = 50

# [ generate initial positions for the agents and respective targets ]
zz_init = np.random.uniform(low=0, high=20, size=(N, d))
target_pos = np.random.uniform(low=0, high=20, size=(N, d))
print("Initial Positions: {}\tShape: {}".format(zz_init, zz_init.shape))
print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))


# [ create graph ]
# args = {'edge_probability': 0.65, 'seed': seed}
# graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.ERDOS_RENYI, args)
graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.PATH)

# [ define ell_i ]
gamma_1 = np.ones(N)    # equally distributed weights
gamma_2 = np.ones(N)    # equally distributed weights

def generate_launch_description():
    node_list = []

    for i in range(N):
        neighbors_i = np.where(adj[i] > 0)[0].tolist()
        # print(f"neighbors_{i}: {neighbors_i}")

        adj_i = [ adj[i, j] for _, j in enumerate(neighbors_i)]        
        # print(f"adj_{i}: {adj_i}")

        node_list.append(
            Node(
                package="dacs_project",
                namespace=f"agent_{i}",     # different agents
                executable="agent",         # same executable,
                parameters=[                # provide parameters to the agents
                    {
                        "id": i,
                        "neighbours": neighbors_i,
                        "max_iter": max_iter,
                        "adj_i": adj_i,
                        "N": N,
                        # "d": d,
                        "zz_init": zz_init[i].tolist(),
                        "target_pos": target_pos[i].tolist(),
                        "gamma_1": float(gamma_1[i]),
                        "gamma_2": float(gamma_2[i]),
                        "alpha": float(alpha),
                    }
                ],
                output="screen",
                # prefix=f'xterm -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )

    return LaunchDescription(node_list)
