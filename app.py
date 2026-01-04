import tkinter as tk
from data_file import *
import customtkinter as ctk
from pygame_handler import *
import matplotlib.pyplot as plt
from genetic_algo import Bot, x_screen, y_screen

# --- UI Configuration ---
# Supported modes : Light, Dark, System (System follows OS theme)
ctk.set_appearance_mode("System")  
 
# Supported themes : green, dark-blue, blue
ctk.set_default_color_theme("blue")   
 
# Dimensions of the main application window
appWidth, appHeight = 670, 400

# --- Simulation Initialization ---
# Initialize the Gameplay handler with a default bot and an empty environment
game = Gameplay(bot=Bot(start_pt=start_pt, end_pt=end_pt, obst_bounds=np.array([])))
# Reset bot state with zero gains and initial heading
game.bot_sim.load(np.zeros((3, 2, game.bot_sim.num_vars)), np.pi)

class App(ctk.CTk):
    """
    The main Dashboard class. Handles button clicks, dropdown menus, 
    and coordinates between the simulation and data storage.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("GENETIC ALGORITHM SIMULATOR")  
        self.geometry(f"{appWidth}x{appHeight}")
        self.resizable(0,0) # Fixed window size for layout consistency
        
        # Internal state for PID gains
        self.pid = np.zeros((3, 2, game.bot_sim.num_vars))
        iter = 0 # Helper variable for grid-based layout positioning

        # --- Obstacle Management Section ---
        new_obst_button = ctk.CTkButton(self, text="Get new Obstacles", command=self.new_obst)
        new_obst_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
        iter += 1

        load_obst_button = ctk.CTkButton(self, text="Load Obstacles", command=self.load_obst)
        load_obst_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")
        
        self.obst_menu = ctk.CTkComboBox(self, values=self.get_files_list(file_path_obst))
        self.obst_menu.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter += 1
        
        # --- GA Weights (PID) Section ---
        load_pid_button = ctk.CTkButton(self, text="Load PID values", command=self.load_pid)
        load_pid_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")
        
        self.pid_menu = ctk.CTkComboBox(self, values=self.get_files_list(file_path_weights))
        self.pid_menu.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter += 1

        # --- Data Visualization Section (Fitness/Probability Heatmaps) ---
        load_fit_prob_button = ctk.CTkButton(self, text="Load Fitness/Probability", command=self.imshow_graph)
        load_fit_prob_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")

        self.graph_menu_1 = ctk.CTkComboBox(self, values=self.get_files_list(file_path_temp))
        self.graph_menu_1.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter += 1

        # --- Path History Section (Replaying Simulations) ---
        load_pos_rec = ctk.CTkButton(self, text="Load Position Record", command=self.plot_path)
        load_pos_rec.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")

        self.graph_menu_2 = ctk.CTkComboBox(self, values=self.get_files_list(file_path_temp))
        self.graph_menu_2.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter += 1
        
        # --- Main Action Buttons ---
        simulate_button = ctk.CTkButton(self, text="Simulate", command=self.simulate)
        simulate_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        iter += 1

        refresh_button = ctk.CTkButton(self, text="Refresh", command=self.refresh)
        refresh_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        iter += 1

        # Sidebar Textbox for logging or displaying PID table data
        self.displayBox = ctk.CTkTextbox(self, width=300, height=300) 
        self.displayBox.grid(row=0, column=3, columnspan=3, rowspan=10, padx=10, pady=10)        
    
    def get_files_list(self, file_path):
        """Scans specified directories and returns a list of files for the dropdowns."""
        try:
            files = os.listdir(file_path)
            files.append("None")
            return files[::-1] # Reverse to keep newest or 'None' accessible
        except FileNotFoundError:
            return ["None"]

    def new_obst(self):
        """Triggers Pygame to draw obstacles and saves the result to a .npy file."""
        dialog = ctk.CTkInputDialog(text="Enter File Name:", title="Save Obstacles")
        input_name = dialog.get_input()
        if input_name:
            file_name = f"{input_name}.npy"
            game.get_obst() # Opens Pygame interaction window
            save_files(file_path=file_path_obst, file_name=file_name, val=np.array(game.bot_sim.obst_bounds))
    
    def load_obst(self):
        """Loads a pre-defined set of obstacle boundaries from the selected file."""
        file_name = self.obst_menu.get()
        if file_name != "None":
            game.bot_sim.obst_bounds = load_files(file_path=file_path_obst, file_name=file_name)
        else:
            game.get_obst(new=0) # Resets to default boundary if None

    def load_pid(self):
        """Loads trained PID gains and prepares the bot simulation with those weights."""
        file_name = self.pid_menu.get()
        if file_name != "None":
            self.pid = load_files(file_path=file_path_weights, file_name=file_name)
        game.bot_sim.load(pid=self.pid)

    def simulate(self):
        """Starts the real-time Pygame simulation with current obstacles and PID gains."""
        game.simulate(max_steps=max_steps, dt_val=dt)

    def refresh(self):
        """Updates all dropdown menus to reflect newly created or deleted files."""
        self.obst_menu.configure(values=self.get_files_list(file_path_obst))
        self.pid_menu.configure(values=self.get_files_list(file_path_weights))
        self.graph_menu_1.configure(values=self.get_files_list(file_path_temp))
        self.graph_menu_2.configure(values=self.get_files_list(file_path_temp))

    def imshow_graph(self):
        """Visualizes training logs (like fitness trends) as a heatmap using Matplotlib."""
        file_name = self.graph_menu_1.get()
        if file_name != "None":
            val = load_files(file_path=file_path_temp, file_name=file_name)
            plt.figure(figsize=(8, 6))
            plt.imshow(val, aspect='auto')
            plt.title(f"Fitness Evolution: {file_name}")
            plt.xlabel("Population Index")
            plt.ylabel("Generation")
            plt.colorbar(label='Fitness Score')
            plt.show()

    def plot_path(self):
        """Replays the movement history of a specific generation in the Pygame window."""
        file_name = self.graph_menu_2.get()
        if file_name != "None":
            # Load path records (uses pickle flag=0 because records are list objects)
            path_rec = load_files(file_path=file_path_temp, file_name=file_name, flag=0)
            game.init_pygame()
            for gen in path_rec:
                # Iteratively draw the positions for each generation
                if not game.draw_pos_record(gen):
                    break # Stop if Pygame window is closed
            game.quit_pygame()

if __name__ == "__main__":
    app = App()
    app.mainloop()
