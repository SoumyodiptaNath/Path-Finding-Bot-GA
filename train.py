from tqdm import tqdm
from data_file import *
from genetic_algo import *
from datetime import datetime


if __name__ == "__main__":
    file_name = "PATH 2.npy"
    obst_bounds = load_files(file_path=file_path_obst, file_name=file_name)
    
    #INIT AND TRAIN
    gen_pop = Genetic_Population(num_var=num_var, range_var=range_var, levels_var=levels_var, 
                                population_size=population_size, start_pt=start_pt, end_pt=end_pt, 
                                max_steps=max_steps, num_bots=num_bots, elitist=True, obst_bounds=obst_bounds)

    train_pos_rec = []
    for i in tqdm(range(num_gen)):
        train_pos_rec.append(gen_pop.reproduce())
        
    #GET AND SAVE VALUES
    optimal_pid = gen_pop.population[gen_pop.elite_id]()
    now = datetime.now().strftime(rf"%m_%d_%Y %H_%M")

    save_files(file_path=file_path_weights, file_name=f"{now} {population_size}_{num_gen}.npy", val=optimal_pid)
    save_files(file_path=file_path_temp, file_name=f"{now} fitness.npy", val=np.array(gen_pop.fitness_history))
    save_files(file_path=file_path_temp, file_name=f"{now} probability.npy", val=np.array(gen_pop.prob_history))
    save_files(file_path=file_path_temp, file_name=f"{now} train_pos_rec.bin", val=train_pos_rec, flag=0)

    print("\nDONE SAVING!!!\n")
 