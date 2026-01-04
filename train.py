from tqdm import tqdm
from data_file import *
from genetic_algo import *
from datetime import datetime

"""
TRAINING SCRIPT: GA-PID Optimization
This script performs the heavy lifting of evolving the PID gains. 
It runs offline (without visualization) to maximize performance.
"""

if __name__ == "__main__":
    # --- Environment Setup ---
    # Load a specific obstacle configuration for the bots to train on
    file_name = "PATH 2.npy"
    obst_bounds = load_files(file_path=file_path_obst, file_name=file_name)
    
    # --- Initialization ---
    # Setup the population with the hyperparameters defined in data_file.py
    # Elitism is enabled to ensure the best performing bot is never lost between generations
    gen_pop = Genetic_Population(
        num_var=num_var, 
        range_var=range_var, 
        levels_var=levels_var, 
        population_size=population_size, 
        start_pt=start_pt, 
        end_pt=end_pt, 
        max_steps=max_steps, 
        num_bots=num_bots, 
        elitist=True, 
        obst_bounds=obst_bounds
    )

    # --- Training Loop ---
    # train_pos_rec stores the movement history of every bot in every generation
    train_pos_rec = []
    print(f"Starting Evolution: {num_gen} generations...")
    
    # tqdm provides a progress bar in the terminal
    for i in tqdm(range(num_gen)):
        # .reproduce() evaluates the current population and creates the next generation
        train_pos_rec.append(gen_pop.reproduce())
        
    # --- Data Persistence ---
    # Extract the PID gains from the best-performing individual (the elite)
    optimal_pid = gen_pop.population[gen_pop.elite_id]()
    
    # Generate a unique timestamp for the file names to avoid overwriting previous runs
    now = datetime.now().strftime(rf"%m_%d_%Y %H_%M")

    # 1. Save the best PID weights for future simulation
    save_files(file_path=file_path_weights, 
               file_name=f"{now} {population_size}_{num_gen}.npy", 
               val=optimal_pid)
    
    # 2. Save training logs for performance visualization (Fitness and Selection Probability)
    save_files(file_path=file_path_temp, 
               file_name=f"{now} fitness.npy", 
               val=np.array(gen_pop.fitness_history))
    
    save_files(file_path=file_path_temp, 
               file_name=f"{now} probability.npy", 
               val=np.array(gen_pop.prob_history))
    
    # 3. Save the full position records for replaying the evolution in the GUI
    save_files(file_path=file_path_temp, 
               file_name=f"{now} train_pos_rec.bin", 
               val=train_pos_rec, 
               flag=0) # Use flag=0 for pickle storage of nested lists

    print(f"\nTraining Complete. Files saved with timestamp: {now}\n")
