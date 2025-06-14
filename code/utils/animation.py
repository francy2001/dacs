import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_targets(ax, target_pos):
    targets_plot = ax.scatter(
        target_pos[:, 0],
        target_pos[:, 1],
        marker="x",
        s=120,
        color="tab:blue",
        label="Targets"
    )
    return targets_plot

def draw_communication_graph(ax, zz, adj):
    NN = zz.shape[0]

    for ii in range(NN):
        for jj in range(NN):
            if adj[ii, jj] > 0 & (jj > ii): # is it an egde? plot only one "side"
                communication_plot = ax.plot(
                    [zz[ii, 0], zz[jj, 0]],
                    [zz[ii, 1], zz[jj, 1]],
                    linewidth=1,
                    color="steelblue",
                    linestyle="solid",
                    # label="Comm. graph"
                )
    return communication_plot
    
def draw_agents(ax, zz, adj, vip_idx=None, safety_distance=None):
    # zz: all the agents positions at the current iteration
    agents_plot = ax.plot(
        zz[:, 0], # all agents' component x
        zz[:, 1], # all agents' component y
        marker="o",
        markersize=15,
        linestyle='none',
        fillstyle="full",
        color="tab:red",
        label="Agents",
        zorder=2
    )

    if vip_idx is not None:
        ax.plot(
            zz[vip_idx, 0],
            zz[vip_idx, 1],
            marker="o",
            markersize=15,
            linestyle='none',
            fillstyle="full",
            color="tab:cyan",
            label="VIP Agent",
            zorder=3

        )
    
    if safety_distance is not None:
        patches = [plt.Circle((x, y), safety_distance, color='r', alpha=0.75, zorder=1) for x,y in zz]
        for p in patches:
            ax.add_patch(p)

    plots = (agents_plot)

    if adj is not None:
        newtork_plot = draw_communication_graph(ax, zz, adj)
        plots = (agents_plot, newtork_plot)
    
    return plots

def draw_trajectories(ax, zz):
    trajectories_plot = ax.plot(
        zz[:, :, 0], # [iteration, who, state-component]
        zz[:, :, 1],
        color="tab:gray",
        linestyle="dashed",
        alpha=0.5,
        # label="Trajectories"
    )
    return trajectories_plot

def draw_barycenter(ax, zz):
    barycentre_plot = ax.plot(
        np.mean(zz[:, 0]),
        np.mean(zz[:, 1]),
        marker="^",
        markersize=15,
        linestyle='none',
        fillstyle="none",
        color="tab:green",
        label="Barycenter"
    )
    return barycentre_plot


def animation(ax, zz, target_pos, adj=None, wait_time=0.001, skip=1, vip_idx=None, title="Animation", safety_distance=None):
    max_iter = zz.shape[0] # time horizon len
    axes_lim = (np.min(zz) - 1, np.max(zz) + 1)
    
    ax.set_xlim(axes_lim)
    ax.set_ylim(axes_lim)
    ax.axis("equal")
    ax.set_xlabel("first component")
    ax.set_ylabel("second component")
    
    plt.gcf().canvas.manager.set_window_title(title)

    for tt in range(0, max_iter, skip):
        ax.cla()
        traj = draw_trajectories(ax, zz)
        targ = draw_targets(ax, target_pos)
        bary = draw_barycenter(ax, zz[tt])
        agen = draw_agents(ax, zz[tt], adj, vip_idx, safety_distance)

        # ax.legend([traj, targ, bary, agen], ["traj", "targ", "bary", "agen"])
        ax.legend(numpoints=1)
        ax.set_title(f"{title} \n Iteration counter = {tt}")
        ax.figure.canvas.draw()
        
        if plt.waitforbuttonpress(wait_time):
            break
