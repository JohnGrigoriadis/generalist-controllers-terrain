
from biped_terrain import BipedalWalker
from network import NeuralNetwork, fill_parameters

import numpy as np
import time
import json
import torch
import pickle

from evotorch.algorithms import XNES
from evotorch.neuroevolution import NEProblem

from joblib import Parallel, delayed

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
        self.terrain_params = self.select_terrains(terrain_params)

        self.keep_terrains = self.terrain_params.copy()

        self.max_fitness = max_fitness

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU !!!")
        else:
            print("Using CPU")
        self.net.to(self.device)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.ter_num = 0

        self.good_terrains = []
        self.bad_terrains = []

        self.generalists = []
        self.gen_terrains = []

    def evaluation_function(self, net: NeuralNetwork):

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

        return total_reward

    def run(self, generations = 1000, pSize = 25, sigma = 0.1):

        print("Running EVO")

        problem = NEProblem(
            objective_sense="max",
            network_eval_func=self.evaluation_function,
            network=NeuralNetwork(self.state_size, 20, self.action_size).eval(),
            num_actors= 0, # Number of parallel evaluations.
            initial_bounds=(-0.00001, 0.00001)
                )

        searcher = XNES(
            problem,
            stdev_init = sigma,
            popsize = pSize
                )
        
        print("")
        print(f"Population: {searcher._popsize}, Generations: {generations}")
        print("")

        save = True
        count = 0
        count_split = 0

        for gen in range(generations):
            searcher.step()
            fitness = searcher.status["best"].evals[0].item()

            print(f"Generation: {gen}, Best Fitness: {fitness:.3f}")

            # Replace if statement with modulo calculation
            self.ter_num = (self.ter_num + 1) % len(self.terrain_params) 

            if fitness >= self.max_fitness and self.ter_num == 0:
                
                good, bad, reached_goal, avg_fitness = self.split(searcher.status["best"].values)

                # This is a backup in case the code fails later on
                save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/XNES_BipedWalker_ens_{count_split}.pt"
                torch.save(searcher.status["best"].values, save_path)
                count_split += 1

                self.terrain_params = bad.copy()  # make the terrains only the bad ones + 1 good for generalization
                self.gen_terrains.append(good) # Save the terrains in an adjacent list for later merging

                net = NeuralNetwork(24, 20, 4)
                fill_parameters(net, searcher.status["best"].values)

                new_problem = NEProblem(
                    objective_sense="max",
                    network_eval_func=self.evaluation_function,
                    network=net.eval(), # Hopefully the initial weights are the same as the best individual
                    num_actors= 0, # Number of parallel evaluations.
                    initial_bounds=(-0.00001, 0.00001)
                        )

                # I will  initialize a new searcher for the new set of terrains.
                searcher = XNES(
                    new_problem,
                    stdev_init = sigma,
                    popsize = pSize
                        )

                
                if len(bad) == 0:
                    print("No terrains left, stopping the evolution.")
                    print(f"Generation: {gen}, Final Best Fitness: {fitness:.3f}, Avg Fitness: {avg_fitness:.3f}")
                    break
        
        if "bad" in locals():
            if len(bad) > 0:
                # self.bad_terrains = bad.copy()
                print(f"Terrains that could not be solved: {bad}")

        self.merge_generalists(self.generalists)

        return searcher, self.generalists

    def split(self, best):
        """
        Test the best individual on all terrains and split them into Good and Bad terrains.
        Then continue the evolution on the Bad terrains to encourage generalization.

        If the best individual reaches the target fitness on 85% of the terrains, the initial goal is reached,
        but the evolution continues to hopefully find a better generalist controller.
        """

        # Fill in the parameters of the best individual
        fill_parameters(self.net, best)

        max_steps = 1600
        s = 0

        reached_goal = False

        good, bad, avg_score = [], [], []
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

            if score >= self.max_fitness: # Be a bit more lenient with accepting good and bad terrains.
                good.append(param)
            else:
                bad.append(param)

            avg_score.append(score)

        if len(good) > int(len(self.terrain_params) * 0.85):
          reached_goal = True


        self.env.close()

        print(f"  Evaluation: Good terrains: {len(good)}, Bad terrains : {len(bad)}")

        return good, bad, reached_goal, np.mean(avg_score)
    
    def select_terrains(self, terrains) -> list:
        """
        This fucntions selects 10 terrains from each subgroup to do the evolution 
        This way the samples are representive of the total terrains
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
            filepath = "generalist-controllers-terrain\XNES_Biped\Experiment_Results\Generalists\generalists_dict.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump({0: self.gen_terrains[0]}, f)

        else:

            # Create a matrix with all fitnesses for each generalist on each terrain
            gen_matrix = []
            
            for i, generalist in enumerate(generalists):
                fill_parameters(self.net, generalist)
                controller_fit = Parallel(n_jobs=4)(delayed(self.evaluation_function)(self.net, params) for params in self.keep_terrains)
                gen_matrix.append(controller_fit)

            # Select the best generalist controller for each terrain, with no overlap
            generalists_new = {i: [] for i in range(len(generalists))}
            # ter_indeces = [i for i in range(len(gen_terrains[0]))]

            gen_matrix_T = np.array(gen_matrix).T
            for ter_idx, fits in enumerate(gen_matrix_T):

                best_fit = max(list(fits))
                contr = list(fits).index(best_fit)
                generalists_new[contr].append(self.keep_terrains[ter_idx])
            

            filepath = "generalist-controllers-terrain\XNES_Biped\Experiment_Results\Generalists\generalists_dict.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump(generalists_new, f)

            print("Merging complete, generalists saved to file.")


def experiment():

    # Load the json file biped_exp.json
    with open('generalist-controllers-terrain/XNES_Biped/biped_exp.json') as file:
        data = json.load(file)

    start = time.time()

    env = BipedalWalker() # Hardcore is off by default

    state_size, hidden_size, action_size = data["NN-struc"]
    net = NeuralNetwork(state_size, hidden_size, action_size)

    noise_range = data["noise"] # [0.0, 1.1]  I want to include 1.0
    slope_range = data["slope"] # [-0.5, 0.5] Slope 0.5 is already pretty steep
    step_size = data["step_size"] # 0.1

    terrain_params = generate_terrain(noise_range, slope_range, step_size) # 120 terrains with the current bounds and step size

    sigma = data["stdev_init"]
    pSize = data["population"]  # At the moment the population is set manually at 30, but can be set chosen automatically by XNES (23)
    generations = 500 #data["generations"]
    target_fitness = data["targetFitness"]

    evo = EVO(env, net, terrain_params, target_fitness)
    searcher, generalists = evo.run(generations = generations, pSize = pSize, sigma = sigma)

    end = time.time()
    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/{data['filename']}.pt"
    torch.save(searcher.status["best"].values, save_path)

    return searcher, generalists

if __name__ == "__main__":
    searcher, generalists = experiment()

    # print(searcher.status["best"].values, searcher.status["best"].evals)

    # if generalists != []:
    for i, generalist in enumerate(generalists):
        torch.save(generalist, f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/Generalists/generalist_0_{i}.pt")
