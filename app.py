import tkinter as tk
from data_file import *
import customtkinter as ctk
from pygame_handler import *
import matplotlib.pyplot as plt
from genetic_algo import Bot, x_screen, y_screen

# Supported modes : Light, Dark, System
ctk.set_appearance_mode("System")  
 
# Supported themes : green, dark-blue, blue
ctk.set_default_color_theme("blue")   
 
# Dimensions of the window
appWidth, appHeight = 670, 400

# Init Bot and Gameplay
game = Gameplay(bot=Bot(start_pt=start_pt, end_pt=end_pt, obst_bounds=np.array([])))
game.bot_sim.load(np.zeros((3, 2, game.bot_sim.num_vars)), np.pi)
 

class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("GENETIC ALGORITH SIMULATOR")  
        self.geometry(f"{appWidth}x{appHeight}")
        self.resizable(0,0)
        
        self.pid = np.zeros((3, 2, game.bot_sim.num_vars))
        iter = 0

        new_obst_button = ctk.CTkButton(self, text="Get new Obstacles", command=self.new_obst)
        new_obst_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
        iter+=1

        load_obst_button = ctk.CTkButton(self, text="Load Obstacles", command=self.load_obst)
        load_obst_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")
        
        self.obst_menu = ctk.CTkComboBox(self, values=self.get_files_list(file_path_obst))
        self.obst_menu.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter+=1
        
        load_pid_button = ctk.CTkButton(self, text="Load PID values", command=self.load_pid, )
        load_pid_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")
        
        self.pid_menu = ctk.CTkComboBox(self, values=self.get_files_list(file_path_weights))
        self.pid_menu.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter+=1

        load_fit_prob_button = ctk.CTkButton(self, text="Load Fitness/Probabilty", command=self.imshow_graph)
        load_fit_prob_button.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")

        self.graph_menu_1 = ctk.CTkComboBox(self, values=self.get_files_list(file_path_temp))
        self.graph_menu_1.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter+=1

        load_pos_rec = ctk.CTkButton(self, text="Load Position Record", command=self.plot_path)
        load_pos_rec.grid(row=iter, column=1, columnspan=1, padx=10, pady=10, sticky="ew")

        self.graph_menu_2 = ctk.CTkComboBox(self, values=self.get_files_list(file_path_temp))
        self.graph_menu_2.grid(row=iter, column=0, columnspan=1, padx=10, pady=10, sticky="ew")
        iter+=1
        
        simulate_button = ctk.CTkButton(self, text="Simulate", command=self.simulate)
        simulate_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        iter+=1

        refresh_button = ctk.CTkButton(self, text="Refresh", command=self.refresh)
        refresh_button.grid(row=iter, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        iter+=1

        self.displayBox = ctk.CTkTextbox(self, width=300, height=300) 
        self.displayBox.grid(row=0, column=3, columnspan=3, rowspan=10, padx=10, pady=10)        
        # self.displayBox.insert("0.0", wlcm_text)
        # self.displayBox.delete("0.0", "end")
        # self.displayBox.insert("0.0", "")
    
    
    def get_files_list(self, file_path):
        files = os.listdir(file_path)
        files.append("None")
        return files[::-1]


    def new_obst(self):
        dialog = ctk.CTkInputDialog(text="Enter File Name:", title="Test")
        file_name = f"{dialog.get_input()}.npy"
        game.get_obst()
        save_files(file_path=file_path_obst, file_name=file_name, val=np.array(game.bot_sim.obst_bounds))
    
    
    def load_obst(self):
        file_name = self.obst_menu.get()
        if file_name != "None":
            game.bot_sim.obst_bounds = load_files(file_path=file_path_obst, file_name=file_name)
        else:
            game.get_obst(new=0)

    
    def load_pid(self):
        file_name = self.pid_menu.get()
        if file_name != "None":
            self.pid = load_files(file_path=file_path_weights, file_name=file_name)
        game.bot_sim.load(pid=self.pid)


    def simulate(self):
        game.simulate(max_steps=max_steps, dt_val=dt)


    def refresh(self):
        self.obst_menu.configure(values=self.get_files_list(file_path_obst))
        self.pid_menu.configure(values=self.get_files_list(file_path_weights))
        self.graph_menu_1.configure(values=self.get_files_list(file_path_temp))
        self.graph_menu_2.configure(values=self.get_files_list(file_path_temp))


    def imshow_graph(self):
        file_name = self.graph_menu_1.get()
        if file_name != "None":
            val = load_files(file_path=file_path_temp, file_name=file_name)
            plt.imshow(val)
            plt.title(file_name)
            plt.xlabel("Population")
            plt.ylabel("Generation")
            plt.show()


    def plot_path(self):
        file_name = self.graph_menu_2.get()
        if file_name != "None":
            path_rec = load_files(file_path=file_path_temp, file_name=file_name, flag=0)
            game.init_pygame()
            for gen in path_rec:
                if not game.draw_pos_record(gen):
                    break
            game.quit_pygame()


if __name__ == "__main__":
    A = App()
    A.mainloop()