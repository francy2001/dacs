# DACS: Distributed Autonomous Cooperative Systems

This repository contains the implementation of two distributed optimization algorithms applied to multi-agent coordination problems. The work is part of a university project focused on cyber-physical systems and distributed control, particularly:

- **Task 1**: Gradient Tracking for cooperative localization
- **Task 2**: Aggregative Tracking for distributed aggregative optimization (also implemented in ROS2)

---

## :file_folder: Project Structure

```
.
├── code
│   ├── task_1                          # Cooperative localization algorithms
│   │   ├── cooperative_localization.py
│   │   └── gradient_tracking.py
│   ├── task_2                          # Aggregative tracking with ROS2 integration
│   │   ├── aggregative_tracking.py
│   │   ├── docker_ws                   # Dockerized ROS2 workspace
│   │   │   └── das_ros2_ws
│   │   │       └── src/dacs_project
│   │   │           ├── dacs_project   # Core agent logic and visualization
│   │   │           │   ├── agent.py
│   │   │           │   └── visualizer.py
│   │   │           ├── launch_folder  # Parametric ROS2 launch file
│   │   │           ├── resource       # RViz configuration and metadata
│   │   │           ├── setup.py       # ROS2 Python package setup
│   │   │           └── test           # Linting and formatting checks
│   │   └── safety_controller.py       # Safety layer for collision avoidance
│   └── utils                          # Common utilities (plotting, graph)
│       ├── animation.py
│       ├── graph_utils.py
│       └── plot_utils.py
└── README.md
```

---

## :rocket: Getting Started

> Requirements:
- Python 3.8+
- ROS 2 (Humble or compatible) if running Task 2
- Docker (for containerized ROS2 environment)
- Recommended: `matplotlib`, `numpy`, `networkx`

### :test_tube: Task 1: Cooperative Localization
Run directly from `code/task_1`.

To test the gradient tracking implementation run:
```bash
python gradient_tracking.py
```

For the cooperative localization run:
```bash
python cooperative_localization.py
```

### :robot: Task 2: Aggregative Tracking (ROS2)

For the aggregative tracking implementation:
```bash
python aggregative_tracking.py
```

To test the aggregative with safety controller run:
```bash
python safety_controller.py
```

Inside `code/task_2/docker_ws`, build and run using Docker.

Launch simulation via:
```bash
ros2 launch dacs_project dacs_parametric_launch.py
```

---

## :brain: About

This project explores distributed control and coordination strategies in multi-agent systems. It integrates theoretical algorithmic design with practical implementation in simulation environments.

---

## :page_facing_up: License

This project is released under the MIT License.

---

## :mailbox_with_mail: Contact

For questions or collaboration, feel free to open an issue or contact [@francy2001](https://github.com/francy2001).
