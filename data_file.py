import sys
import pickle
import os.path
import numpy as np
from prettytable import PrettyTable

# --- File Path Configurations ---
# Dynamically locate directories for temporary data, obstacles, and neural/PID weights
file_path_temp = os.path.join(sys.path[0], "FILES", "TEMP_DATA", "")
file_path_obst = os.path.join(sys.path[0], "FILES", "OBSTACLES", "")
file_path_weights = os.path.join(sys.path[0], "FILES", "WEIGHTS", "")


def save_files(file_path, file_name, val, flag=1):
    """
    Saves data to disk. 
    - flag=1: Uses NumPy (.npy) for efficient numerical storage.
    - flag=0: Uses Pickle for general Python objects.
    """
    with open(f"{file_path}{file_name}", 'wb') as f_out:
        if flag: np.save(f_out, val)
        else: pickle.dump(val, f_out)


def load_files(file_path, file_name, flag=1):
    """
    Loads data from disk.
    - flag=1: Loads NumPy arrays.
    - flag=0: Loads Pickled objects.
    """
    with open(f"{file_path}{file_name}", 'rb') as f_in:
        if flag: return np.load(f_in)
        else: return pickle.load(f_in)


def show_table(pid):
    """
    Formats and prints the PID gain matrices into readable tables.
    Separate tables are generated for Linear Velocity (v) and Angular Velocity (w).
    """
    final_pid = np.copy(pid)
    num_params = final_pid.shape[-1]
    optimal_pid = np.around(final_pid, decimals=5)

    # Initialize tables for Linear (v) and Angular (w) control components
    pid_table_v = PrettyTable(["Variable", "P", "I", "D"])
    pid_table_w = PrettyTable(["Variable", "P", "I", "D"])

    for i in range(num_params):
        # Handle LIDAR sensor inputs
        if i < num_params - 2:
            pid_table_v.add_row(["LIDAR " + str(i+1), optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["LIDAR " + str(i+1), optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        
        # Handle Position Deviation (Distance to goal)
        elif i == num_params - 2:
            pid_table_v.add_row(["POS DEV", optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["POS DEV", optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        
        # Handle Angular Deviation (Heading error to goal)
        else:
            pid_table_v.add_row(["ANG DEV", optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["ANG DEV", optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        
    print("\n--- Linear Velocity (V) PID Gains ---")
    print(pid_table_v)
    print("\n--- Angular Velocity (W) PID Gains ---")
    print(pid_table_w)


# --- Genetic Algorithm Hyperparameters ---
num_var = 30           # Number of variables to optimize
num_gen = 25           # Total generations for evolution
num_bots = 48          # Number of bot instances for parallel evaluation
population_size = 48   # Total individuals in the population

# Search space: Gains are initialized within [-0.1, 0.1]
range_var = 0.1 * np.ones((num_var, 2))
range_var[:,0] = -range_var[:,0]

# Chromosome resolution: 5 bits per variable (32 discrete levels per gain)
levels_var = 5 * np.ones(num_var, dtype="int")

# --- Bot & Environment Parameters ---
dt = 0.01              # Simulation timestep
x_screen = 900         # Window width
y_screen = 600         # Window height
max_steps = 2000       # Timeout limit for each bot simulation
start_pt = np.array([200.0, 500.0])
end_pt = np.array([700.0, 100.0])
