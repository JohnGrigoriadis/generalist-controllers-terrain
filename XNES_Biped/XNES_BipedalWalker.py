
from biped_terrain import BipedalWalker

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import time
import json

from evotorch.algorithms import XNES
from evotorch.neuroevolution import NEProblem

# Neural network
class NeuralNetwork(nn.Module):
    """
    3 layer neural network:
        - Input layer: 24 nodes
        - Hidden layer: 20 nodes
        - Output layer: 4 nodes

        - Activation function: Tanh
    """

    def __init__(self, state_size, hidden_size, action_size):
        super(NeuralNetwork, self).__init__()  # Call the parent class constructor
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_size)
        
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()

    
    def forward(self, state):
        state = torch.Tensor(state)  # Convert state to a Tensor
        x = self.act1(self.layer1(state))
        x = self.act2(self.layer2(x))
        return x

# net = NeuralNetwork(24, 4)
        
# Generate terrains to test the generalist controller on
def generate_terrain(noise_range, slope_range, step_size):
    noise_values = np.arange(noise_range[0], noise_range[1], step_size)
    slope_values = np.arange(slope_range[0], slope_range[1], step_size)

    terrains = np.array(np.meshgrid(noise_values, slope_values)).T.reshape(-1, 2)

    # I could shuffle the terrains to make the evolution more interesting. But I will keep them in order for now.

    return terrains

class EVO():
    '''
    With the normal environment, the total_reward goal is 300, but since the varying terrains make the environment harder, 
    I will set the goal to 250.
    
    Parameters:
    - env: BipedalWalker environment
    - net: NeuralNetwork
    - terrain_params: List of terrain parameters (noise, slope)
    '''

    def __init__(self, env : BipedalWalker, net : NeuralNetwork, terrain_params, max_fitness = 250):
        self.env = env
        self.net = net
        self.terrain_params = terrain_params
        self.max_fitness = max_fitness
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        

        self.ter_num = 17

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
            network_eval_func= self.evaluation_function,
            network=self.net, 
            num_actors= 0, # Number of parallel evaluations.
            initial_bounds=(-0.00001, 0.00001)
                )
            
        searcher = XNES(
            problem,
            stdev_init = sigma,
            popsize = pSize  
                )
        print(searcher._popsize)

        save = True

        for gen in range(generations):
            searcher.step()
            fitness = searcher.status["best"].evals 

            if gen % 100 == 0:
                print(f"Generation: {gen}, Best Fitness: {round(fitness[0].item(), 3)}")

            self.ter_num +=1
            if self.ter_num == len(self.terrain_params):
                self.ter_num = 0
            
            # Save the first individual that reaches a fitness of 250 
            # More or less a fail-safe to keep at least one good individual, in case the evolution fails after a certain point.
            if fitness[0].item() >= self.max_fitness and save:
                save_path = "generalist-controllers-terrain\XNES_Biped\XNES_BipedWalker_250.pt"
                torch.save(searcher.status["best"].values, save_path)
                save = False
                
        return searcher


def experiment():
    """
    Runs the experiment and evolution process.
    """
    # load the json file biped_exp.json
    with open('biped_exp.json') as f:
        data = json.load(f)

    start = time.time()

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
    generations = data["generations"]
    target_fitness = data["targetFitness"]
    
    evo = EVO(env, net, terrain_params, target_fitness)
    searcher = evo.run(generations = generations, pSize = pSize, sigma = sigma)
    
    end = time.time()
    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    save_path = f"generalist-controllers-terrain\XNES_Biped\{data['filename']}.pt"
    torch.save(searcher.status["best"].values, save_path)

    return searcher

if __name__ == "__main__":
    searcher = experiment()

    print(searcher.status["best"].values, searcher.status["best"].evals)
