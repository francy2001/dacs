# Distributed Autonomous Cooperative Systems (DACS) :duck:

This repository contains the final submission for the Distributed Autonomous Systems (DAS) project, developed during the 2024/2025 academic year at the University of Bologna.

The project explores how distributed optimization can be leveraged to coordinate teams of autonomous agents, such as mobile robots, without relying on centralized control. In modern cyber-physical networks, tasks like localization, coordination, and safe navigation must often be achieved under strict constraints on communication, computation, and privacy. Distributed algorithms provide a scalable and robust solution to these challenges.

We study two main distributed optimization paradigms:

- **Consensus Optimization**: where agents cooperatively agree on a common decision variable (e.g., shared target location).
- **Aggregative Optimization**: where agents balance their own objectives with a collective term, such as the team barycenter.

Both algorithms are validated in simulation, and the aggregative scenario is also implemented in ROS2 with an additional safety controller to ensure collision-free coordination.

---

## :open_file_folder: Folder Structure

```
.
├── README.md               # High-level overview (this file)
├── report_group_16.pdf     # Final report (compiled PDF)
├── report/                 # LaTeX source files and figures
│   └── figs/
├── code/                   # Source code for both tasks
│   ├── task_1/             # Gradient tracking for cooperative localization
│   ├── task_2/             # Aggregative tracking and safety controller
│   └── README.md           # Instructions and structure for the code
```

---

## :rocket: Summary of Tasks

### :small_blue_diamond: Task 1: Cooperative Localization via Consensus Optimization
- Implements a **Distributed Gradient Tracking** algorithm.
- Agents estimate a shared target location based on noisy local measurements.
- Convergence to the global optimum is validated across different network topologies.

### :small_orange_diamond: Task 2: Distributed Aggregative Optimization
- Agents minimize local cost functions dependent on their own state and a team barycenter.
- Includes a **ROS2-based implementation** using publish/subscribe communication.
- A **Control Barrier Function (CBF)**-based safety layer ensures real-time collision avoidance.
- Simulations explore behaviors with different agent priorities and team cohesion levels.

---

## :page_facing_up: License

This project is distributed under the MIT License.

---

## :busts_in_silhouette: Authors

- Francesca Bocconcelli  
- Luca Fantini
