import sys
import pickle
import os.path
import numpy as np
from prettytable import PrettyTable

file_path_temp = os.path.join(sys.path[0], "FILES", "TEMP_DATA", "")
file_path_obst = os.path.join(sys.path[0], "FILES", "OBSTACLES", "")
file_path_weights = os.path.join(sys.path[0], "FILES", "WEIGHTS", "")


def save_files(file_path, file_name, val, flag=1):
    with open(f"{file_path}{file_name}", 'wb') as f_out:
        if flag: np.save(f_out, val)
        else: pickle.dump(val, f_out)


def load_files(file_path, file_name, flag=1):
    with open(f"{file_path}{file_name}", 'rb') as f_in:
        if flag: return np.load(f_in)
        else: return pickle.load(f_in)


def show_table(pid):
    final_pid = np.copy(pid)
    num_params = final_pid.shape[-1]
    optimal_pid = np.around(final_pid, decimals=5)

    pid_table_v = PrettyTable(["Variable", "P", "I", "D"])
    pid_table_w = PrettyTable(["Variable", "P", "I", "D"])
    for i in range(num_params):
        if i<num_params-2:
            pid_table_v.add_row(["LIDAR " + str(i+1), optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["LIDAR " + str(i+1), optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        elif i == num_params-2:
            pid_table_v.add_row(["POS DEV", optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["POS DEV", optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        else:
            pid_table_v.add_row(["ANG DEV", optimal_pid[0, 0, i], optimal_pid[1, 0, i], optimal_pid[2, 0, i]])
            pid_table_w.add_row(["ANG DEV", optimal_pid[0, 1, i], optimal_pid[1, 1, i], optimal_pid[2, 1, i]])
        
    print(pid_table_v)
    print(pid_table_w)


#GA PARAMS
num_var = 30
num_gen = 25 
num_bots = 48
population_size = 48
range_var = 0.1*np.ones((num_var, 2))
range_var[:,0] = -range_var[:,0]
levels_var = 5*np.ones(num_var, dtype="int")

#BOT PARAMS
dt=0.01
x_screen = 900
y_screen = 600
max_steps = 2000
start_pt = np.array([200.0, 500.0])
end_pt = np.array([700.0, 100.0])
