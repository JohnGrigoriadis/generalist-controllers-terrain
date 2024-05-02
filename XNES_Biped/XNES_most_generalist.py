
from biped_terrain import BipedalWalker
from network import NeuralNetwork

import numpy as np
import time
import json
from joblib import Parallel, delayed

import torch

from evotorch.algorithms import XNES
from evotorch.neuroevolution import NEProblem

# Generate terrains to test the generalist controller on
def generate_terrain(noise_range, slope_range, step_size):
    """
        The terrains are shuffled in a way that the slope changes every generation,
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
    With the normal environment (noise=0.1, slope=0.0), the total_reward goal is 300, but since the varying terrains make the environment harder,
    I will set the goal to 250.

    Parameters:
    - env: BipedalWalker environment
    - net: NeuralNetwork
    - terrain_params: List of terrain parameters (noise, slope)
    '''

    def __init__(self, env : BipedalWalker, net : NeuralNetwork, terrain_params, max_fitness = 250):
        self.env = env
        self.net = net
        self.terrain_params = terrain_params[:5]
        self.max_fitness = max_fitness

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.max_evals = 30 #115_000
        self.evals = 0

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

    def run_simulation(self, net: NeuralNetwork, terrain):

        self.env.noise, self.env.slope = terrain
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


    def evaluation_function(self, net: NeuralNetwork):

        scores = Parallel(n_jobs=4)(delayed(self.run_simulation)(net, terrain) for terrain in self.terrain_params)
        self.evals += len(scores)

        return np.mean(scores)

    def run(self, generations = 1000, pSize = 25, sigma = 0.1):

        print("Running EVO")

        problem = NEProblem(
            objective_sense="max",
            network_eval_func= self.evaluation_function,
            network=NeuralNetwork(24, 20, 4),
            num_actors= None, # Number of parallel evaluations.
            initial_bounds=(-0.00001, 0.00001),
            device=self.device,
            eval_dtype=torch.float32
                )

        searcher = XNES(
            problem,
            stdev_init = sigma,
            popsize = pSize
                )
        
        print(f"Population: {searcher._popsize}, Generations: {generations}, Terrains: {len(self.terrain_params)}")
        print("")

        for gen in range(generations):
            searcher.step()
            fitness = searcher.status["best"].evals[0].item()

            print(f"Generation: {gen}, Best Fitness: {fitness:.3f}")

            if self.evals >= self.max_evals:
                print("Max evaluations reached.\n   Exiting...\n")
                break
            
        return searcher


def experiment():

    # Load the json file biped_exp.json
    with open('XNES_Biped/biped_exp.json') as f:
        data = json.load(f)


    env = BipedalWalker()

    state_size, hidden_size, action_size = data["NN-struc"]
    net = NeuralNetwork(state_size, hidden_size, action_size)

    noise_range = data["noise"] # [0.0, 1.1]  I want to include 1.0
    slope_range = data["slope"]#[-0.5, 0.5] Slope 0.5 is already pretty steep
    step_size = data["step_size"] # 0.1

    terrain_params = generate_terrain(noise_range, slope_range, step_size) # 100 terrains at the moment with the current step size

    sigma = data["stdev_init"]
    # At the moment the population is chosen automatically by XNES (23), but I can set it manually
    pSize = data["population"]
    generations = 250 #data["generations"]
    target_fitness = data["targetFitness"]

    evo = EVO(env, net, terrain_params, target_fitness)

    start = time.time()
    searcher = evo.run(generations = generations, pSize = pSize, sigma = sigma)
    end = time.time()

    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    save_path = f"XNES_Biped/Experiment_Results/Generalists/most_generalist.pt"
    torch.save(searcher.status["best"].values, save_path)

    return searcher

if __name__ == "__main__":
    searcher = experiment()

    print(searcher.status["best"].values, searcher.status["best"].evals)
