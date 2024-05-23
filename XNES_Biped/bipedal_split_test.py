
from biped_terrain import BipedalWalker
from network_old import NeuralNetwork, fill_parameters

import numpy as np
import time
import json
import torch
import pickle

from sklearn.cluster import KMeans

from evotorch.algorithms import XNES
from evotorch.neuroevolution import NEProblem

# from joblib import Parallel, delayed

# Generate terrains to test the generalist controller on
def generate_terrain(noise_range, slope_range, step_size):
    """
    The terrains are generated in a way that the slope changes every generation,
    once all slopes are visited the noise is increased.

    This makes it so the task starts a bit easier in the beginning and it gets
    harder as the noise level increases.
    """

    noise_values = np.arange(noise_range[0], noise_range[1], step_size)
    slope_values = np.arange(slope_range[0], slope_range[1], step_size)

    terrains = np.array(np.meshgrid(noise_values, slope_values)).T.reshape(-1, 2)

    return terrains

def sort_terrains(terrains, A=[0.0, 0.0]):
    """
    Sort the terrains by Euclidian similarity

    Params:
    terrains: list of generated terrain parameters
    A: Mean parameter values of the terrains
    """
    
    norm_ter = terrains / np.max(terrains, axis=0)

    # compute uclidean distance
    euclid_d = []
    for terrain in norm_ter:
        terrain = np.array(terrain)
        temp = A - terrain
        sum_sq = np.dot(temp.T, temp)
        euclid_d.append(sum_sq)


    sorted_args = np.argsort(euclid_d)
    euclid_d = sorted(euclid_d)
    terrains = terrains[sorted_args]

    terrains_below_mean = []
    terrains_above_mean = []
    mean_dist_from_0 = np.mean(euclid_d)
        
    for i in range(len(terrains)):
        if euclid_d[i] < mean_dist_from_0:
            terrains_below_mean.append(terrains[i])
        else:
            terrains_above_mean.append(terrains[i])
            
    return terrains_below_mean, terrains_above_mean

def cluster_terrains(terrains:list):
    """
    Given a set of terrains, cluster them into two groups and return the two clusters.

    Since xNES works better wit similar terrians, so clustering should help in that.

    Params:
    - terrains: list of generated terrain parameters   
    """

    terrains = np.array(terrains)

    if len(terrains) <= 10:
        return terrains, []
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=2, n_init="auto")
    kmeans.fit(terrains)
    labels = kmeans.labels_

    cluster_1 = terrains[labels == 0]
    cluster_2 = terrains[labels == 1]
        
    return cluster_1, cluster_2

class EVO():
    '''
    With the normal environment (noise=0.1, slope=0.0), the total_reward goal is 300,
    but since the varying terrains make the environment harder, I will set the goal to 250.

    Parameters:
    - env: BipedalWalker environment
    - net: NeuralNetwork
    - terrain_params: List of terrain parameters (noise, slope)
    - max_fitness: The target fitness the controller should achieve

    This version of the algorithm holds the reward of the agent for each terrain and split them into Good and Bad.
    Once the target reward is reached *for any terrain*, the evolution is continued in only the Bad terrains.
    '''

    def __init__(self, env : BipedalWalker, net : NeuralNetwork, terrain_params: list, max_fitness = 250):
        self.env = env
        self.net = net
        # self.terrain_params = self.select_terrains(terrain_params)
        # self.terrain_params = terrain_params

        # self.terrain_params
        self.terrain_params, self.bad_terrains,  = sort_terrains(terrain_params) 
        self.keep_terrains = np.concatenate([self.terrain_params, self.bad_terrains])

        self.max_fitness = max_fitness

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {str(self.device).upper()} device for training.")
        self.net.to(self.device)


        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.ter_num = 0

        self.generalists = []
        self.gen_terrains = []

        self.max_evals = 120_000
        self.evals = 0

    def evaluation_function(self, net: NeuralNetwork):
        """
        Evaluate the controller on the current terrain and return the total reward.

        Params:
        - net: NeuralNetwork controller

        Returns:
        - total_reward: Total reward of the network on the current terrain
        """

        self.env.noise, self.env.slope = self.terrain_params[self.ter_num] # Change the terrain in each generation

        obs, _ = self.env.reset()

        done  = False
        nStep = 1600
        total_reward = 0
        s = 0

        while not done :

            action = net.forward(obs).detach().numpy()

            obs, reward, terminated, truncated, _  = self.env.step(action)

            s += 1
            total_reward += reward

            done = terminated or truncated

            if s > nStep:
                break

        self.env.close()
        self.evals += 1

        return total_reward

    def run(self, generations = 5000, pSize = 25, sigma = 0.1):
        """
        Run the evolution algorithm with the given parameters and return a single generalist controller,
        or an ensemble of generalist controllers.
        
        Params:
        - generations: Number of generations
        - pSize: Population size
        - sigma: Initial standard deviation for the search

        Returns:
        - searcher: The XNES searcher object
        - generalists: List of generalist NeuralNetwork controllers
        """

        bad = []

        print("Running EVO")

        problem = NEProblem(
                objective_sense="max",
                network_eval_func=self.evaluation_function,
                network=NeuralNetwork(self.state_size, 20, self.action_size), 
                num_actors= 0, # Number of parallel evaluations.
                initial_bounds=(-0.00001, 0.00001),
                device = "cpu",
                eval_dtype=torch.float32
                    )

        searcher = XNES(
            problem,
            stdev_init = sigma,
             #ranking_method="raw",
            # popsize = pSize
                )
        
        print("")
        print(f"Population: {searcher._popsize}, Generations: {generations:,}, self.max_evals: {self.max_evals:,}")
        print("")

        best_individual = None
        test_net = NeuralNetwork(self.state_size, 20, self.action_size)
        count = 0
        current_best = -np.inf

        fitness_list = {"Performance of best individual" : []}

        for gen in range(generations):

            searcher.step()
            fitness = searcher.status["best"].evals[0].item()
            best_individual = searcher.status["best"].values


            if fitness > current_best:
                current_best = fitness
                improved = searcher.status["iter"]


            # check if the algorithm is stuck
            if gen - improved > (generations / 16) and self.ter_num == 0:
                improved = gen
                self.terrain_params = self.remove_terrain() 
                
            fill_parameters(test_net.eval(), best_individual)
            score = self.evaluation_function(test_net)
            
            # Add the performance of the best individual to a list for plotting.
            fitness_list["Performance of best individual"].append(score)

            noise , slope = self.terrain_params[self.ter_num][0], self.terrain_params[self.ter_num][1]
            print(f"Gen: {gen}, Best Fit: {fitness:.3f}, Gen Best: {score:.3f}, Terrain: [{noise:.1f}, {slope:.1f}]")

            # Replace if statement with modulo calculation
            self.ter_num = (self.ter_num + 1) % len(self.terrain_params) 


            if fitness >= self.max_fitness and self.ter_num == 0:
                
                good, bad = self.split(searcher.status["best"].values)

                self.terrain_params = bad.copy()  # make the terrains only the bad ones + 1 good for generalization
                self.gen_terrains.append(good) # Save the terrains in an adjacent list for later merging
                self.generalists.append(searcher.status["best"].values) # Save the generalist controller

                net = NeuralNetwork(24, 20, 4)
                fill_parameters(net, searcher.status["best"].values)

                torch.save(searcher.status["best"].values, 
                            f"XNES_Biped/Experiment_Results/Generalists/generalist_ter_exp_3_{count}.pt")
                count += 1
                
                new_problem = NEProblem(
                    objective_sense="max",
                    network_eval_func=self.evaluation_function,
                    network=net, 
                    num_actors= 0, # Number of parallel evaluations.
                    initial_bounds=(-0.00001, 0.00001),
                    device = "cpu",
                    eval_dtype=torch.float32
                        )

                # I will  initialize a new searcher for the new set of terrains.
                searcher = XNES(
                    new_problem,
                    stdev_init = sigma,
                        )


                if len(bad) == 0:
                    print("")
                    print("No terrains left, stopping the evolution.")
                    return searcher, self.generalists

            if self.evals >= self.max_evals:
                print("Max evaluations reached.\n   Exiting...")
                break

        if len(bad) > 0:
            print("")
            print(f"{len(bad)} terrains could not be solved: {bad}")

        self.merge_generalists(self.generalists)

        # Write the values of the fitness_list to a file for plotting
        
        return searcher, self.generalists

    def remove_terrain(self):
        """
        Remove a terrain that the controller is stuck on and continue the evolution on the remaining terrains.
        Then add the removed terrain to the bad terrains list.
        """

        print("Removing a terrain")

        to_remove = np.argmax(self.terrain_params, axis=0)[0]
        if len(self.bad_terrains) == 0:
            self.bad_terrains = self.terrain_params[to_remove]
        else:
              self.bad_terrains = np.concatenate((self.bad_terrains, self.terrain_params[to_remove]))
        self.terrain_params = np.delete(self.terrain_params, to_remove, axis=0)

        return self.terrain_params
    
    def split(self, best):
        """
        Test the best individual on all terrains and split them into Good and Bad terrains.
        Then continue the evolution on the Bad terrains to encourage generalization.

        If the best individual reaches the target fitness on 85% of the terrains, the initial goal is reached,
        but the evolution continues to hopefully find a better generalist controller. - NOT USED HERE
        """

        # Fill in the parameters of the best individual
        fill_parameters(self.net.eval(), best)

        max_steps = 1600
        s = 0

        good, bad, avg_score = [], [], []

        if len(self.bad_terrains) > 0:
            self.terrain_params = np.concatenate((self.terrain_params, self.bad_terrains))

        for param in self.terrain_params:

            self.env.noise, self.env.slope = param

            state, _ = self.env.reset()
            done = False
            score = 0
            s = 0

            while not done:
                action = self.net.forward(state).detach().numpy()

                state, reward, terminated, truncated, _ = self.env.step(action)

                score += reward

                if s > max_steps:
                    break

                s += 1

                done = terminated or truncated

            if score >= 220: # Be a bit more lenient with accepting good and bad terrains.
                good.append(param)
            else:
                bad.append(param)

            avg_score.append(score)

        self.env.close()

        bad, self.bad_terrains = cluster_terrains(bad)

        print(f"  Evaluation: Good terrains: {len(good)}, Bad terrains : {len(bad) + len(self.bad_terrains)}, Avg score: {np.mean(avg_score):.3f}")
        if np.mean(avg_score) >= 220:
            print("Average score of controller is above the target, stopping the evolution.")
            return bad, []
        
        print(f"Continueing evolution on {len(bad)} terrains")

        return good, bad 
    
    def select_terrains(self, terrains) -> list:
        """
        This fucntions selects 10 terrains from each subgroup to do the evolution 
        This way the samples are representive of the total terrains

        It is not used anymore, but it is kept for reference.
        """

        # Assuming terrains is a list of terrain objects
        positive_slope, zero_slope, negative_slope = [], [], []

        for terrain in terrains:
            slope = terrain[1]
            if slope > 0.1:
                positive_slope.append(tuple(terrain))
            elif slope > -0.2:
                zero_slope.append(terrain)
            else:
                negative_slope.append(terrain)

        positive_slope = np.array(positive_slope)
        zero_slope = np.array(zero_slope)
        negative_slope = np.array(negative_slope)

        pos_idx = np.random.randint(0, len(positive_slope), size=10)
        zero_idx = np.random.randint(0, len(zero_slope), size=10)
        neg_idx = np.random.randint(0, len(negative_slope), size=10)


        pos_sample = [positive_slope[i] for i in pos_idx]
        zero_sample = [zero_slope[i] for i in zero_idx]
        neg_sample = [negative_slope[i] for i in neg_idx]

        terrains_new = neg_sample + zero_sample + pos_sample
        terrains_new = np.array(terrains_new)

        return terrains_new
    
    def merge_generalists(self, generalists) -> None:
        """
        Evaluate the generalists on all terrains and select the best generalist controller
        for each terrain.

        Params:
        - generalists: List of generalist controllers
        - gen_terrians: List of terrains for each generalist controller
        """

        if len(generalists) == 0:
            print("No generalists to merge, stopping the process.")
            
        
        elif len(generalists) == 1:
            print("Only one generalist, merging not possible, saving it to file.")
            filepath = "XNES_Biped/Experiment_Results\Generalists\generalist_single_dict.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump({0: self.gen_terrains[0]}, f)

        else:

            # Create a matrix with all fitnesses for each generalist on each terrain
            gen_matrix = []
            
            for i, generalist in enumerate(generalists):
                fill_parameters(self.net, generalist)

                controller_fit = []
                self.terrain_params = self.keep_terrains.copy()

                for num in range(len(self.keep_terrains)):
                    self.ter_num = num
                    controller_fit.append(self.evaluation_function(self.net))

                gen_matrix.append(controller_fit)

            # Select the best generalist controller for each terrain, with no overlap
            generalists_new = {i: [] for i in range(len(generalists))}

            gen_matrix_T = np.array(gen_matrix).T
            for ter_idx, fits in enumerate(gen_matrix_T):

                best_fit = max(list(fits))
                if best_fit >= self.max_fitness:
                    contr = list(fits).index(best_fit)
                    generalists_new[contr].append(self.keep_terrains[ter_idx])
            

            filepath = "XNES_Biped/Experiment_Results\Generalists\generalists_dict_1.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump(generalists_new, f)

            print("Merging complete, generalists saved to file.")


def experiment():
    """
    Runs the experiment with the given parameters and returns the best controller and generalist controllers.
    """

    # Load the json file biped_exp.json
    with open('XNES_Biped/biped_exp.json',"r") as file:
        data = json.load(file)

    start = time.time()

    env = BipedalWalker() # Hardcore is off by default

    state_size, hidden_size, action_size = data["NN-struc"]
    net = NeuralNetwork(state_size, hidden_size, action_size)

    noise_range = data["noise"] # [0.0, 0.7]  
    slope_range = data["slope"] # [-0.5, 0.5] Slope 0.5 is already pretty steep
    step_size = data["step_size"] # 0.1

    terrain_params = generate_terrain(noise_range, slope_range, step_size) # 120 terrains with the current bounds and step size
    # terrain_params = sort_terrains(terrain_params) # Sort the terrains by Euclidian similarity


    sigma = data["stdev_init"]
    pSize = data["population"]  # At the moment the population is set manually at 30, but can be set chosen automatically by XNES (23)
    generations = data["generations"]
    target_fitness = data["targetFitness"]

    evo = EVO(env, net, terrain_params, target_fitness)
    searcher, generalists = evo.run(generations = generations, pSize = pSize, sigma = sigma)

    end = time.time()
    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    return searcher, generalists

if __name__ == "__main__":
    searcher, generalists = experiment()

    # print(searcher.status["best"].values, searcher.status["best"].evals)


    for i, generalist in enumerate(generalists):
        torch.save(generalist, f"XNES_Biped/Experiment_Results/Generalists/generalist_ter_3_{i}.pt")
