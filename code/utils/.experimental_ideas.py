# -------------------------------------------------
# |        ANIMATION - traj. diff colors ]        |
# -------------------------------------------------
def draw_trajectories(ax, zz):
    N = zz.shape[1] # N
    cmap = plt.colormaps['rainbow']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, (N)))
    for i, color in enumerate(colors):
        # ax.plot([0, i], color=color)
        trajectories_plot = ax.plot(
            zz[:, i, 0], # [iteration, who, state-component]
            zz[:, i, 1],
            color=color,
            linestyle="dashed",
            alpha=0.5,
            # label="Trajectories"
        )
    return trajectories_plot

# ---------------------------------------------------
# |          FEISTEL MAP POPULATION                 |
# ---------------------------------------------------
"""A simple 8-round Feistel network to encode/decode 8-bit integers.
See https://en.wikipedia.org/wiki/Feistel_cipher for construction details.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

ROUNDS = 8

def init(seed):
    global KEYS
    # set seed for reproducibility of the generation
    random.seed(seed)
    KEYS = [(random.randint(11, 19), random.randint(1, 200)) for _ in range(ROUNDS)]

def encode(number):
    """Feistel encryption operation to map 8-bit inputs to 8-bit outputs."""
    left, right = number >> 4, number & 0x0f
    for keynum in range(ROUNDS):
        left, right = right, left ^ feistel(right, keynum)
    return left << 4 | right


def decode(number):
    """Inverse (decryption) Feistel operation."""
    left, right = number >> 4, number & 0x0f
    for keynum in reversed(range(ROUNDS)):
        left, right = right ^ feistel(left, keynum), left
    return left << 4 | right


def feistel(number, keynum):
    """Feistel non-invertible transformation function,"""
    offset, multiplier = KEYS[keynum]
    return (number + offset) * multiplier & 0x0f


if __name__ == '__main__':
    N = 75
    seed = 42
    init(seed)

    spawn_points = np.zeros((2**4, 2**4))
    plt.figure(figsize=(4, 4))

    for i in range(N):
        print(f"i:{i} --> encode(i):{encode(i)}")
        enc = encode(i)
        x, y = enc >> 4, enc & 0x0f
        spawn_points[x,y] = 1

    plt.imshow(spawn_points, cmap='binary')
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    plt.show()  

# In the cooperative_localization.py or in the aggregative_tracking.py
# [ divide the grid in 16x16=256 cells, populate it without collisions using feistel encryption ]
scale = 1
robot_pos = np.zeros(shape=(N, d))
target_pos = np.zeros(shape=(Nt, d))

def populate_grid_map_randomly():
    # encodes n in range [0, N-1]
    for i, n in zip(range(N), range(N)):
        enc = feistel.encode(n)
        x, y = enc >> 4, enc & 0x0f
        robot_pos[i] = np.array([x, y]) * scale
    
    # encodes n in range [N, N+Nt-1]
    for i, n in zip(range(Nt), range(N, N+Nt)):
        enc = feistel.encode(n)
        x, y = enc >> 4, enc & 0x0f
        target_pos[i] = np.array([x, y]) * scale

populate_grid_map_randomly()

print("Robot Positions: {}\tShape: {}".format(robot_pos, robot_pos.shape))
print("Target Positions: {}\tShape: {}".format(target_pos, target_pos.shape))


# -------------------------------------------------
# |               TASK 1.2 - z_init               |
# -------------------------------------------------
# TODO: educated guess? maybe choose the starting position of the target 
# on a sphere of radius noisy_distances[i] centered in robot_pos[i].
# z_init = np.random.uniform(size=(N, Nt, d))


# -------------------------------------------------
# |            BIRKHOFF VON NEUMANN               |
# -------------------------------------------------
# general case of a graph with birkhoff-von-neumann weights
graph, weighted_adj = graph_utils.create_graph_birkhoff_von_neumann(N, 5)


# -------------------------------------------------
# |           SIMULATE COLLISION AVOIDANCE        |
# -------------------------------------------------
# [ collision avoidance ]
def sim_collision_avoidance():
    gamma_1 = np.ones(N)
    gamma_2 = np.ones(N)
    cost_functions = []
    for i in range(N):
        cost_functions.append(lambda zz, barycenter, i=i: cost_fn(zz, target_pos[i], barycenter, gamma_1[i], gamma_2[i]))
    
    z_init = np.zeros(shape=(N,d))
    target_pos = np.zeros(shape=(N,d))
    # (0) O -->  ~  --> X (3)
    # (1) O -->  ~  --> X (2)
    # (2) O -->  ~  --> X (1)
    # (3) O -->  ~  --> X (0)
    for i in range(N):
        z_init[i] = np.array([0, 2*i])
        target_pos[i] = np.array([15, (2*(N-1))-2*i])
    delta = 0.5 # in real coordinates          
    __sim_aggregative_tracking("Collision avoidance", z_init, target_pos, dim, cost_functions, gamma_1, gamma_2, graph_type=graph_utils.GraphType.COMPLETE, safety_distance=delta)
        