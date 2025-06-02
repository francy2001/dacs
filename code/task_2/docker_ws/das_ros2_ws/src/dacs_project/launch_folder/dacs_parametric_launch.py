from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import numpy as np
from utils import graph_utils

seed = 38
np.random.seed(seed)

N = 3
d = 2
graph, adj = graph_utils.create_graph_with_metropolis_hastings_weights(N, graph_utils.GraphType.PATH)
max_iter = 10

zz_init = np.random.uniform(low=0, high=20, size=(N, d))

def generate_launch_description():
    node_list = []

    for i in range(N):
        neighbors_i = np.where(adj[i] > 0)[0].tolist()
        # print(f"neighbors_{i}: {neighbors_i}")
        
        adj_i = [ adj[i, j] for pos, j in enumerate(neighbors_i)]
        # print(f"adj_{i}: {adj_i}")

        node_list.append(
            Node(
                package="dacs_project",
                namespace=f"agent_{i}",     # different agents
                executable="agent",         # same executable,
                parameters=[                # provide parameters to the agents
                    {
                        "id": i,                        # int
                        "neighbours": neighbors_i,      # list of ints
                        "max_iter": max_iter,           # int
                        "adj_i": adj_i,                 # list of floats
                        # "d": d,                         # int
                        "zz_init": list(zz_init[i])     # list of floats
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}" -fg white -bg black -fs 12 -fa "Monospace" -hold -e',
            )
        )

    return LaunchDescription(node_list)
