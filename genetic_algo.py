import math
import numpy as np
from icecream import ic
import concurrent.futures
from data_file import x_screen, y_screen

class Genetic_Controller():
    """
    Represents an individual in the genetic population. 
    Encapsulates the chromosome (binary representation) and decodes it into PID gains.
    """
    def __init__(self, num_var, range_var, levels_var, number_variables, state_angles):
        self.elite = False
        self.num_var = num_var
        self.levels_var = levels_var
        self.state_angles = state_angles
        
        # Scaling parameters to map binary values to physical PID ranges
        self.var_shift = range_var[:,0]
        self.var_scale = range_var[:,1] - range_var[:,0]
        
        self.number_variables = number_variables
        self.cumulative_levels = np.cumsum(levels_var)
        
        # Initialize random binary chromosome
        self.chromosomes = np.random.RandomState().randint(2, size=(self.cumulative_levels[-1]))

    def __call__(self):
        """Decodes binary chromosomes into a 3x2xN PID gain matrix."""
        start = 0
        values = np.zeros(self.num_var)
        
        # Binary to Decimal conversion
        for i in range(self.num_var):
            binary = self.chromosomes[start:self.cumulative_levels[i]]
            decimal = binary.dot(1 << np.arange(binary.shape[-1] - 1, -1, -1))
            values[i] = decimal/(2**self.levels_var[i]-1)
            start = self.cumulative_levels[i]

        # Rescale values to specified range
        values = self.var_shift + self.var_scale*values
        values = values.reshape((3, 2, -1))
        
        # Construct PID matrix: [Proportional, Integral, Derivative] x [Linear, Angular] x [States]
        PID = np.zeros((3, 2, self.number_variables))
        
        # Symmetry handling for sensor angles
        PID[:,:,:self.state_angles//2+1] = values[:,:,:self.state_angles//2+1]
        PID[:,:,self.state_angles//2+1:self.state_angles] = -np.flip(values[:,:,:self.state_angles//2], axis=-1)
        PID[:,:,self.state_angles:] = values[:,:,self.state_angles//2+1:]
        
        # Constraint: Integral gain for the middle sensor angle is zeroed to prevent drift
        PID[:,1,self.state_angles//2] = 0
        return PID

    def mutate(self, mut_prob=0.1):
        """Performs bit-flip mutation based on mutation probability."""
        if self.elite == False:
            arr = np.random.RandomState().rand(self.cumulative_levels[-1])
            mut = (arr <= mut_prob).astype('int')
            # Flip bits using XOR
            self.chromosomes = np.bitwise_xor(mut, self.chromosomes)

class Genetic_Population():
    """
    Manages a population of Genetic_Controllers, handles selection, 
    crossover, and parallel evaluation.
    """
    def __init__(self, num_var, range_var, levels_var, population_size, 
                 obst_bounds, start_pt, end_pt, max_steps, num_bots, elitist=False):
        
        self.elite_id = 0
        self.elitist = elitist
        self.num_var = num_var
        self.num_bots = num_bots
        self.max_steps = max_steps
        self.total_levels = np.sum(levels_var)
        self.population_size = population_size
        self.cumulative_levels = np.cumsum(levels_var)
        
        if population_size % num_bots != 0:
            raise ValueError("Population size must be divisible by number of bots")
            
        # Initialize bots and their corresponding controllers
        self.bot_army = [Bot(start_pt, end_pt, obst_bounds) for i in range(self.num_bots)]
        self.population = [Genetic_Controller(num_var, range_var, levels_var, 
                           self.bot_army[0].num_vars, self.bot_army[0].state_angles) 
                           for i in range(population_size)]
        
        self.prob_history = []
        self.fitness_history = []

    def evaluate(self, member, bot_num):
        """Simulates a single bot and returns its fitness score."""
        pid_values = member()
        self.bot_army[bot_num].load(pid_values)
        fitness, pos_record = self.bot_army[bot_num].simulate(max_steps=self.max_steps, dt=0.01)
        return fitness, pos_record, bot_num

    def evaluate_population(self):
        """Evaluates the entire population using multi-processing."""
        fitness = np.zeros(self.population_size)
        gen_pos_record = []
        
        # Process in batches equal to the number of available bot instances
        for batch in range(self.population_size//self.num_bots):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(self.evaluate, self.population[batch*self.num_bots+i], i) 
                           for i in range(self.num_bots)]

                for f in concurrent.futures.as_completed(results):
                    curr_fitness, pos_record, i = f.result()
                    fitness[batch*self.num_bots+i] = curr_fitness
                    gen_pos_record.append(pos_record)

        self.fitness_history.append(list(fitness))
        return fitness, gen_pos_record

    def reproduce(self):
        """Main GA loop: Selection, Crossover, and Mutation."""
        fitness, gen_pos_rec = self.evaluate_population()
        
        # Normalize fitness for probability distribution (Roulette Wheel Selection)
        prob_fitness = fitness - np.min(fitness) 
        prob = prob_fitness/np.sum(prob_fitness)
        self.prob_history.append(list(prob))

        # Elitism: Preserve the best performing individual
        if self.elitist == True:
            self.population[self.elite_id].elite = False
            index = np.argmax(fitness)
            self.elite_id = index
            self.population[index].elite = True
        
        # Select offspring indices based on fitness probability
        offsprings_index = np.random.choice(np.arange(self.population_size, dtype="int"), 
                                            self.population_size, replace=True, p=prob)
        
        if self.elitist:
            offsprings_index[self.elite_id] = self.elite_id
            
        offsprings = self.get_chromosomes()[offsprings_index]
        elite_indices = offsprings_index == self.elite_id
        
        self.set_elite()

        if self.elitist:
            offsprings[self.elite_id] = self.population[self.elite_id].chromosomes
        
        # Apply genetic operators
        offsprings = self.crossover(offsprings, elite_indices)
        self.set_chromosomes(offsprings)
        self.mutate()
        
        return gen_pos_rec

    def crossover(self, population, elite_indices, crossover_prob=1e-1):
        """Performs uniform crossover between pairs of individuals."""
        for i in range(self.population_size//2):
            crossover_vars = np.random.rand(self.num_var)
            crossover_vars = (crossover_vars <= crossover_prob)
            crossover_index = np.zeros(self.total_levels, dtype = 'int')
            
            # Identify bit segments corresponding to variables selected for crossover
            crossover_index[np.where(crossover_vars[np.searchsorted(self.cumulative_levels, 
                            np.arange(self.total_levels, dtype='int'), side='right')])] = 1

            parent_1 = np.copy(population[2*i])
            parent_2 = np.copy(population[2*i+1])

            # Swap bits
            temp = parent_1[crossover_index]
            parent_1[crossover_index] = parent_2[crossover_index]
            parent_2[crossover_index] = temp
            
            # Update population (skip if individual is the elite)
            if not elite_indices[2*i]:
                population[2*i] = np.copy(parent_1)
            if not elite_indices[2*i+1]:
                population[2*i+1] = np.copy(parent_2)
        return population

    def mutate(self):
        """Triggers mutation for each individual in the population."""
        for i in range(self.population_size):
            self.population[i].mutate()

    def get_chromosomes(self):
        c_set = np.zeros((self.population_size, self.total_levels), dtype="int")
        for i, p in enumerate(self.population):
            c_set[i] = p.chromosomes
        return c_set

    def set_chromosomes(self, c_set):
        for i, p in enumerate(self.population):
            p.chromosomes = c_set[i]

    def set_elite(self):
        for i in range(self.population_size):
            self.population[i].elite = i == self.elite_id

class Bot():
    """
    A simulated mobile robot with ray-casting sensors (Lidar-like) 
    and PID-driven kinematics.
    """
    def __init__(self, start_pt, end_pt, obst_bounds, view_range=200, radius=30, vel_max=50, num_angles=8):
        self.int_pts = [] # Intersection points for sensors
        self.radius = radius
        self.thresh = radius # Collision threshold
        self.end_pt = end_pt
        self.start_pt = start_pt
        self.view_range = view_range
        self.obst_bounds = obst_bounds
        
        if num_angles % 2 == 1:
            raise ValueError("Number of angles must be an even number")
        
        self.state_angles = num_angles//2 + 1
        self.iter_angles = self.state_angles - 1
        self.num_vars = self.state_angles + 2

    def load(self, pid, init_angle=np.pi):
        """Resets the bot state and loads new PID gains."""
        self.v = 0.0 # Linear velocity
        self.w = 0.0 # Angular velocity
        self.P, self.I, self.D = pid[0], pid[1], pid[2]
        self.fitness = 0.0
        self.is_safe = True
        self.is_done = False
        self.pos_record = []
        self.dist_record = []
        self.angle = init_angle
        self.pos = np.copy(self.start_pt)
        
        # Initial heading vector
        cosine, sine = np.cos(self.angle), np.sin(self.angle)
        self.dir = np.array([[cosine, -sine], [sine, cosine]])
        self.proximity_goal = np.linalg.norm(self.end_pt - self.start_pt)
        self.curr_state = np.zeros(self.state_angles + 2)
        self.get_state()
        self.prev_state = np.copy(self.curr_state)

    def get_state(self):
        """Ray-casting sensor implementation to detect obstacles."""
        self.int_pts = []
        curr_angle = -self.angle
        del_angle = np.pi/self.iter_angles

        for index in range(self.iter_angles):
            ray_dir = np.array([np.cos(curr_angle), np.sin(curr_angle)])
            min_dist = self.view_range*np.ones(2)
            
            for line_info in self.obst_bounds:
                start, line_dir = line_info[0], line_info[1]
                
                # Solve for intersection between ray and obstacle line segment
                den = line_dir[0]*ray_dir[1] - line_dir[1]*ray_dir[0]
                if den == 0.0: continue # Parallel lines

                del_r = self.pos - start
                t_num = del_r[0]*ray_dir[1] - del_r[1]*ray_dir[0]

                if 0 < t_num <= den or den <= t_num < 0:
                    u = (-line_dir[0]*del_r[1] + line_dir[1]*del_r[0])/den
                    curr_dist = abs(u)
                    int_pt_index = int(u >= 0)
                    if curr_dist <= min_dist[int_pt_index]:
                        min_dist[int_pt_index] = curr_dist

            # Collision check
            if np.any(min_dist < self.thresh):
                self.is_safe = False
                self.fitness -= 5000.0
                return

            if index == 0: 
                self.curr_state[-3] = min_dist[0]
            self.curr_state[index] = min_dist[1]
            curr_angle += del_angle
        
        # Update goal-related states
        del_pos = self.pos - self.end_pt
        del_pos_mag = np.linalg.norm(del_pos)
        self.proximity_goal = del_pos_mag

        if del_pos_mag < self.thresh:
            self.is_done = True
            self.fitness += 2000.0
            return

        self.dist_record.append(del_pos_mag)
        self.pos_record.append(np.copy(self.pos))
        self.curr_state[-2] = del_pos_mag
        # Calculate angular error to goal
        self.curr_state[-1] = np.sign(np.cross(self.dir[1,], del_pos)) * \
                             np.arccos(np.dot(self.dir[1,], del_pos)/del_pos_mag)

    def control_bot(self, dt):
        """Applies the PID control law to update bot kinematics."""
        state_matrix = np.zeros((3, len(self.curr_state)))
        state_matrix[0] = self.curr_state # Proportional term
        state_matrix[1] = (self.curr_state + self.prev_state)/2 # Integral approx
        state_matrix[2] = self.curr_state - self.prev_state # Derivative term
        
        # Compute control outputs: [v, w] = P*e + I*sum(e) + D*de/dt
        P = self.P @ state_matrix[0]
        I = self.I @ state_matrix[1]
        D = self.D @ state_matrix[2]
        
        self.v, self.w = P + I + D
        self.angle += self.w*dt
        self.pos += self.dir[1,]*self.v*dt

        # Normalize angle to [-2pi, 2pi]
        if abs(self.angle) > 2*math.pi:
            self.angle -= math.copysign(2*math.pi, self.angle)

        cosine, sine = np.cos(self.angle), np.sin(self.angle)
        self.dir = np.array([[cosine, -sine], [sine, cosine]])

    def fitness_func(self, iter):
        """Calculates fitness based on distance to goal, safety, and time."""
        fitness = 1000.0 - 0.1*np.mean(np.array(self.dist_record)) + \
                  self.fitness - 0.5*iter*self.is_safe - self.proximity_goal
        return fitness

    def simulate(self, max_steps, dt):
        """Runs the simulation for a fixed number of steps or until termination."""
        iter = 0
        running = True
        while running and iter < max_steps:
            iter += 1
            self.control_bot(dt)
            self.prev_state = np.copy(self.curr_state)
            self.get_state()
            if not self.is_safe or self.is_done:
                running = False

        fitness = self.fitness_func(iter)
        return fitness, np.array(self.pos_record)
