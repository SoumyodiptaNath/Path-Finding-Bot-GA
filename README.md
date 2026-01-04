# 🤖 GA-PID Autonomous Navigator
### *Optimizing PID Controllers using Genetic Algorithms for Robotic Navigation*



This project implements a **Genetic Algorithm (GA)** to evolve a high-dimensional **PID Controller** for a mobile robot. The robot uses 2D ray-casting (Lidar-like sensors) to navigate through user-defined obstacle courses. The system includes a multi-processed training suite, a real-time Pygame visualizer, and a modern `customtkinter` dashboard.

---

## 🛠 Features
* **Genetic Optimization:** Evolves gains for Linear and Angular PID control across multiple sensor inputs.
* **Parallel Evaluation:** Uses `concurrent.futures` to simulate entire populations across CPU cores simultaneously.
* **Interactive Environments:** Draw custom obstacles in real-time with mouse clicks.
* **Data Visualization:** Heatmaps for fitness evolution and replay features for generational history.
* **Modern GUI:** A centralized dashboard to manage weights, obstacles, and simulations.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `genetic_algo.py` | **Core Logic.** Contains the Genetic Algorithm operations and Robot Kinematics. |
| `pygame_handler.py` | **Visualization.** Manages the Pygame window, rendering, and manual obstacle creation. |
| `gui_main.py` | **Interface.** The `customtkinter` dashboard for managing the simulator. |
| `data_file.py` | **Config.** Global hyperparameters, file paths, and PID table formatting. |
| `main.py` | **Trainer.** The entry point for running the evolution loop offline. |

---

## 🚀 Usage Guide

### 1. Training the Bot
Run the training script to evolve a new set of weights. This runs without a GUI to ensure maximum simulation speed.
```bash
python main.py
```

The script will save .npy weights and .bin position records in the FILES/ directory.2. Launching the SimulatorOpen the GUI Dashboard to interact with the evolved bots and test their performance.Bashpython gui_main.py
Get New Obstacles: Opens a Pygame window. Left Click to start a wall, Right Click to finish it. Press 'Q' to save.Load PID values: Select a timestamped weight file from the dropdown.Simulate: Watch the bot navigate using the loaded weights.🧠 Technical Deep DiveThe ControllerThe controller is a state-dependent PID. Instead of a single error input, it takes a state vector $\mathbf{s}$ consisting of:$N$ distances from ray-casting sensors.Distance to goal (Position Deviation).Heading error (Angular Deviation).Genetic OperationsChromosome: Binary encoded string representing concatenated P, I, and D gains.Selection: Roulette Wheel Selection (Fitness Proportionate).Crossover: Uniform crossover with variable probability to exchange genetic traits.Mutation: Bit-flip mutation to maintain genetic diversity and prevent local minima.🔧 Class & Function ReferenceBot (genetic_algo.py)FunctionArgumentsDescriptionload()pid, init_angleResets bot state and injects new PID matrices.get_state()NoneExecutes ray-casting and updates sensor/goal data.control_bot()dtCalculates $v$ and $w$ using the PID control law.Genetic_Population (genetic_algo.py)FunctionArgumentsDescriptionevaluate_population()NoneDistributes individuals to bot instances for parallel simulation.reproduce()NoneThe Selection $\rightarrow$ Crossover $\rightarrow$ Mutation loop.Gameplay (pygame_handler.py)FunctionArgumentsDescriptionget_obst()newOpens the UI for manual obstacle placement.draw_pos_record()genReplays the path of every individual in a generation.📈 Data VisualizationYou can visualize the "Learning" of your algorithm by loading the fitness logs. A heatmap represents the fitness of the entire population across time (generations).📦 RequirementsTo install the necessary dependencies, run:Bashpip install numpy pygame customtkinter matplotlib prettytable tqdm icecream
Created as part of an exploration into Evolutionary Robotics and PID Control.
---

### Pro-Tip for your Repo
Make sure your folder structure looks like this before you push, otherwise the imports might break:
```text
.
├── main.py
├── gui_main.py
├── genetic_algo.py
├── pygame_handler.py
├── data_file.py
└── FILES/
    ├── TEMP_DATA/
    ├── OBSTACLES/
    └── WEIGHTS/
```
