# import gymnasium as gym
import numpy as np
from typing import List

import torch
import torch.nn as nn
# import torch.nn.functional as F

# import keyboard

from biped_terrain import BipedalWalker 

noise_range = [0.0, 1.1] # I want to include 1.0 
slope_range = [0.0, 1.1] # I want to include 1.0
step_size = 0.2

def generate_terrain(noise_range, slope_range, step_size):
    noise_values = np.arange(noise_range[0], noise_range[1], step_size)
    slope_values = np.arange(slope_range[0], slope_range[1], step_size)

    terrains = np.array(np.meshgrid(noise_values, slope_values)).T.reshape(-1, 2)

    return terrains

# Anil did all this manually, but I think a normal NN structure would work fine.

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()  # Call the parent class constructor
        self.layer1 = nn.Linear(state_size, 20)  
        self.layer2 = nn.Linear(20, action_size)  
        self.tanh = nn.Tanh()

    def forward(self, state):
        state = torch.Tensor(state) # Convert state to a Tensor and 
        # print(state)
        x = self.tanh(self.layer1(state))
        x = self.tanh(self.layer2(x))
        return x

    def get_weights(self):
        return [self.layer1.weight, self.layer2.weight]
    
    def set_weights(self, weights):
        self.layer1.weight = nn.Parameter(weights[0])
        self.layer2.weight = nn.Parameter(weights[1])
    

# Test to make sure that the NN and weights are working
# network = NeuralNetwork(10, 4) # for test purposes
# weights = network.get_weights()
# weights


class XNES_Replica():
    """
    This class creates mutations of the weights of the neural network and then tests them on one tenvironment.
    Then selects the best performing mutation and returns the weights of that mutation on the next environment.
    This happens as many times as the variable generations.

    At the end we should have a list of weights that perform good on all environments at the same time.

    Params:
    pop_num: the number of mutations to make
    env: the environment to test on 
    network: the neural network used to take actions and make mutations 
    generations: the number of generaions to run the algorithm for (once all environments have been tested on, we start from the first environment again)

    """
    
    def __init__(self, pop_num : int, env : BipedalWalker, network : NeuralNetwork, terrains: List[List[float]], generations : int=1008) -> None:
        self.pop_num = pop_num
        self.env = env
        self.network = network
        self.terrains = terrains 
        self.generations = generations

        self.population = []
        self.ter_num = 0
        self.best_reward_per_gen = None


    def mutate(self, weights: List[List[torch.Tensor]]) -> None:
        """
        This function takes in the weights of the network and creates a list with the new mutated population.
        """

        new_pop = []

        for _ in range(self.pop_num):
            new_pop.append([weight + torch.randn_like(weight) for weight in weights])

        self.population = new_pop


    def run_simulation(self, weights: List[List[torch.Tensor]]) -> float:
        """
        This function runs one simulation of the environment with the given weights and returns the reward.
        """

        self.env.noise, self.env.slope = self.terrains[self.ter_num] 
        state, _ = self.env.reset() 
        self.network.set_weights(weights)
        total_reward = 0
        while True:
            action = self.network(state).cpu().numpy()
            state, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        
        return total_reward

    def run_and_return_best(self, population) -> List[torch.Tensor]:
        """
        This function runs one simulation for each mutation and returns the weights of the best performing mutation.
        """

        rewards = []
        for individual in population:
            rewards.append(self.run_simulation(individual))
        
        self.ter_num += 1
        if self.ter_num == len(self.terrains):
            self.ter_num = 0
        
        self.best_reward_per_gen = max(rewards)
        return population[np.argmax(rewards)]
    
    def main(self) -> None:
        """
        This function runs the whole algorithm for the given number of generations..
        
        Process:
        retrieve initial weights
        start loop

        mutate weights
        get best performing weights (run_and_return_best) on terrain 1

        mutate weights
        get best performing weights (run_and_return_best) on terrain 2

        ...
        """

        # Get initial weights
        weights = self.network.get_weights()
        weights = self.run_and_return_best(weights)

        # Start loop
        for gen in range(self.generations - 1):
            # Mutate weights
            self.mutate(weights)

            # Get best performing weights
            weights = self.run_and_return_best(self.population)

            if gen % 100 == 0:
                print(f"Generation: {gen}")
                print(f"Best reward: {self.best_reward_per_gen}")
                print("")

        return weights


env = BipedalWalker()

state_size, action_size = 24, 4
network = NeuralNetwork(state_size, action_size)


terrain_params = generate_terrain(noise_range, slope_range, step_size) # 36 terrains at the moment with the current step size
# print(terrain_params.shape) 

population_size = 10
generations = 5
xnes = XNES_Replica(population_size, env, network, terrain_params, generations)

# Lets hope this works
generalist_weights = xnes.main()

torch.save(generalist_weights, 'best_weights.pt')
