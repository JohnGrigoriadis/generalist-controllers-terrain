
import numpy as np
from typing import List

import torch
import torch.nn as nn
# import torch.nn.functional as F

import keyboard as kb # Used to stop the simulation manually if needed
import time # Used to time the simulation only during testing

from biped_terrain import BipedalWalker 


"""
Things to do:
1. Find why the simulations don't stop running
    a. prob because i wasn't incrementing s # fixed
2. See if the noise added to the weights is correct (maybe ther eis a better way to do it)
3. The first simulation seems to be in terrain 2, not terrain 1. Why? #fixed
4. At the end of the experiment, all weights are saved. This is slow and not very space efficient. Maybe save only the best weights?
5. Save the good and bad terrains?
6. The NN outputs stay the same after some time. This makes the agent "freeze" and not move. Maybe add some noise to the output of the NN?
    a. Adding bias to the layers of the NN made the performance better. 
    b. Good results for 100 generations. Must test for more generations.
7. Reward from experiments does not match the reward from the environment when rendering afterwards. Why?

Notes to self:
1. O(g,p) worst case time complexity is about 25 seconds for g=5 and p=10
2. Muation noise is best between 0.01 and 0.1. 0.1 is the best for now.

"""

def generate_terrain(noise_range, slope_range, step_size):
    noise_values = np.arange(noise_range[0], noise_range[1], step_size)
    slope_values = np.arange(slope_range[0], slope_range[1], step_size)

    terrains = np.array(np.meshgrid(noise_values, slope_values)).T.reshape(-1, 2)

    return terrains

# A normal NN structure would work fine.

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()  # Call the parent class constructor
        self.layer1 = nn.Linear(state_size, 20)
        self.layer2 = nn.Linear(20, action_size)
        self.tahn = nn.Tanh()
        self.bias1 = nn.Parameter(torch.zeros(20))  # Bias for layer 1
        self.bias2 = nn.Parameter(torch.zeros(action_size))  # Bias for layer 2

    def forward(self, state):
        state = torch.Tensor(state)  # Convert state to a Tensor
        x = self.tahn(self.layer1(state) + self.bias1)  # Add bias to layer 1
        x = self.tahn(self.layer2(x) + self.bias2)  # Add bias to layer 2
        return x

    def get_weights(self):
        return [self.layer1.weight, self.layer1.bias, self.layer2.weight, self.layer2.bias]

    def set_weights(self, weights):
        self.layer1.weight = nn.Parameter(weights[0])
        self.layer1.bias = nn.Parameter(weights[1])
        self.layer2.weight = nn.Parameter(weights[2])
        self.layer2.bias = nn.Parameter(weights[3])
    

# Test to make sure that the NN and weights are working
# network = NeuralNetwork(10, 4) # for test purposes
# weights = network.get_weights()
# weights


class EVO():
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
        self.best_per_gen = []
        self.steps = 1600 # The number of steps to run the simulation for
        self.memory = [] # Used to store the weights of the last few generations.

        # Using GPU will speed up the process. So I hope it works.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")


    def mutate(self, weights: List[torch.Tensor]) -> None:
        """
        This function takes in the weights of the network and creates a list with the new mutated population.
        """

        new_pop = []

        for _ in range(self.pop_num):
            new_pop.append([torch.Tensor(weight.detach().numpy() + np.random.uniform(-0.1, 0.11, size=weight.shape)) for weight in weights])


        self.population = new_pop

        # Also add the original weights to the population
        self.population.append(weights)

       

    def run_simulation(self, weights: List[torch.Tensor]) -> float:
        """
        This function runs one simulation of the environment with the given weights and returns the reward.
        """

         
        state, _ = self.env.reset()
        self.network.set_weights(weights)
        total_reward = 0
        s = 0
        while True:
            state = torch.FloatTensor(state).to(self.device)
            action = self.network.forward(state).detach().cpu().numpy()

            state, reward, done, truncated, _ = self.env.step(action)

            total_reward += reward
            
            # Increment the number of steps
            s += 1

            # If the goal is reached, the simulation is truncated or the number of steps is reached, the simulation stops
            if done or truncated or kb.is_pressed('esc') or s > self.steps:
                break

            # If the user presses 'q' and 'l' at the same time, the entire experiment stops
            # failsafe to stop the simulation
            if (kb.is_pressed('q') and kb.is_pressed("l")):
                exit()
        
        return total_reward
    

    def run_and_return_best(self, population : List[torch.Tensor], gen:int) -> List[torch.Tensor]:
        """
        This function runs one simulation for each mutation and returns the weights of the best performing mutation.
        """

        rewards = []
        for individual in population:
            rewards.append(self.run_simulation(individual))


        return [population[np.argmax(rewards)], max(rewards)]
    

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
        weights, reward = self.run_and_return_best([weights], 100)
        self.memory = [weights, reward]
 
        # Start loop
        for gen in range(self.generations + 1):

            # Mutate weights and make the mutations the new self.population
            self.mutate(weights)


            # Get best performing weights from the new self.population
            weights, reward = self.run_and_return_best(self.population, gen)

            # If the reward is better than the previous best reward, save the weights, else revert back to the previous weights
            if reward > self.memory[-1]:
                self.memory = [weights, reward]
            else:
                weights = self.memory[0]

            # Increment the terrain number
            self.ter_num += 1
            if self.ter_num == len(self.terrains):
                self.ter_num = 0

            self.env.noise, self.env.slope = self.terrains[self.ter_num]

            # Print some info to keep an eye on the progress
            if gen % 1 == 0:
                print(f"Generation: {gen}")
                print(f"Best reward of generation: {reward:6f}")
                print(f"Terrain: Slope={env.slope:.2f}, Noise={env.noise:.2f}")
                print("")

        # Save the best performing weights and the reward every 100 generations
        if gen % 10 == 0:
            self.best_per_gen.append([weights, reward])

        return self.best_per_gen

    
if __name__ == "__main__":
    # start = time.time()
    env = BipedalWalker()

    state_size, action_size = 24, 4
    network = NeuralNetwork(state_size, action_size)

    noise_range = [0.0, 1.1] # I want to include 1.0 
    slope_range = [-0.5, 0.5] # I want to include 1.0
    step_size = 0.1

    terrain_params = generate_terrain(noise_range, slope_range, step_size) # 100 terrains at the moment with the current step size


    population_size = 10
    generations = 100
    xnes = EVO(population_size, env, network, terrain_params, generations)

    # Lets hope this works
    generalist_weights = xnes.main()
    
    # end = time.time()
    # print(f"Time taken: {end - start}")
    
    # print(f"Final reward{generalist_weights[-1]}")
    # torch.save(generalist_weights[-1][0], "generalist-controllers-terrain/Terrain Gen/best_weights.pt")
