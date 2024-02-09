
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import keyboard as kb # Used to stop the simulation manually if needed
# import time # Used to time the simulation only during testing

from biped_terrain import BipedalWalker 


# Generate terrains to test the generalist controller on
def generate_terrain(noise_range, slope_range, step_size):
    noise_values = np.arange(noise_range[0], noise_range[1], step_size)
    slope_values = np.arange(slope_range[0], slope_range[1], step_size)

    terrains = np.array(np.meshgrid(noise_values, slope_values)).T.reshape(-1, 2)

    # I could shuffle the terrains to make the evolution more interesting. But I will keep them in order for now.
    # Or order them in a diffe

    return terrains

# A normal NN structure would work fine.

class NeuralNetwork(nn.Module):
    """
    3 layer neural network:
    - Input layer: 24 nodes
    - Hidden layer: 20 nodes
    - Output layer: 4 nodes
    - Bias: 1 node
    - Activation function: Tanh

    Current issue: 
    The agent stops moving after a while. After the agent takes the first steps, the input stays the same, thus the output stays the same.
    
    Adding some noise to the input state might help. More definete solution is needed though.
    """

    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()  # Call the parent class constructor
        hidden_size = 20 
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_size)
        self.bias1 = nn.Parameter(torch.zeros(1))
        
        self.tahn = nn.Tanh()

    
    def forward(self, state):
        state = torch.Tensor(state)  # Convert state to a Tensor
        # I have tested relu for the hidden layer and it works fine. But since the output layer is tanh, I will use tanh for the hidden layer as well.
        x = self.tahn(self.layer1(state) + self.bias1)  # Add bias to layer 1
        x = self.tahn(self.layer2(x))
        return x

    # Retrieve the weights of the network to begin the evolution
    def get_weights(self):
        return [self.layer1.weight, self.layer1.bias, self.layer2.weight, self.layer2.bias]

    # Set the weights of the network to test the mutations
    def set_weights(self, weights):
        self.layer1.weight = nn.Parameter(weights[0])
        self.layer1.bias = nn.Parameter(weights[1])
        self.layer2.weight = nn.Parameter(weights[2])


class EVO():
    """
    This class creates mutations of the weights of the neural network and then tests them on one tenvironment.
    Then selects the best performing mutation and returns its weights. Those weights are mutated and tested on the next environment.
    This happens as many times as the variable generations.

    At the end we should have a set of weights that perform well on all environments.

    Params:
    pop_num: the number of mutations for each generation
    env: the environment to test on. Custom BipedalWalker environment
    network: the neural network used to take actions 
    generations: the number of generaions to run the algorithm for (once all environments have been tested on, we start again from the first environment)

    """
    
    def __init__(self, pop_num : int, env : BipedalWalker, network : NeuralNetwork, terrains: List[List[float]], generations : int=1008) -> None:
        
        # Input parameters
        self.pop_num = pop_num
        self.env = env
        self.network = network
        self.terrains = terrains 
        self.generations = generations

        # Variables
        self.population = []
        self.ter_num = 0
        self.best_per_gen = []
        # If the hardcore mode is enabled, the number of steps is increased to 2000
        if env.hardcore:
            self.steps = 2000
        else:
            self.steps = 1600 

        self.memory = [] # Used to store the weights of the last few generations.

        # Using GPU will speed up the process. If not available, use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")


    def mutate(self, weights: List[torch.Tensor]) -> None:
        """
        This method takes in the weights of the network and creates a list with the new mutated population.

        This is a simple method of adding noise. I will test other more complex methods in the future. Hopefully, they wield better results.
        """

        new_pop = []

        for _ in range(self.pop_num):
            new_pop.append([torch.Tensor(weight.detach().numpy() + np.random.uniform(-0.1, 0.11, size=weight.shape)) for weight in weights])


        self.population = new_pop

        # Also add the original weights to the population
        self.population.append(weights)

       

    def run_simulation(self, weights: List[torch.Tensor]) -> float:
        """
        This method runs one simulation on the environment with the given weights and returns the reward.
        """
         
        state, _ = self.env.reset()
        self.network.set_weights(weights)
        total_reward = 0
        s = 0
        while True:

            state = torch.FloatTensor(state).to(self.device)
            
            action = self.network.forward(state).detach().numpy()
            # action += np.random.uniform(-0.1, 0.1, size=action.shape)
            state, reward, terminated, truncated, _ = self.env.step(action)

            # The agent stops moving after a while. Adding some noise to the input state might help. More definete solution is needed though.
            # state += np.random.uniform(-0.05, 0.05, size=state.shape)

            total_reward += reward
            
            # Increment the number of steps
            s += 1

            if self.env.render_mode == "human":
                self.env.render()

            done = terminated or truncated

            # If the goal is reached, the simulation is truncated or the number of steps is reached, the simulation stops
            if done or kb.is_pressed('esc') or s > self.steps:
                break

            # If the user presses 'q' and 'l' at the same time, the entire experiment stops
            # failsafe to stop the simulation
            if (kb.is_pressed('q') and kb.is_pressed("l")):
                exit()
        
        self.env.close()
        
        return total_reward
    

    def run_and_return_best(self, population : List[torch.Tensor]) -> List[torch.Tensor]:
        """
        This method runs a simulation for each mutation and returns the weights and reward of the best performing mutation.
        """

        rewards = []
        for individual in population:
            rewards.append(self.run_simulation(individual))


        return [population[np.argmax(rewards)], max(rewards)]
    

    def main(self) -> None:
        """
        This method runs the whole algorithm for the given number of generations.
        
        Process:

        retrieve initial weights -> start loop

        mutate weights
        get best performing individual on terrain 1

        mutate best weights from last generation
        get best performing weights (run_and_return_best) on terrain 2

        Repeat for all terrains
        """

        # Get initial weights
        weights = self.network.get_weights()
        weights1, reward1 = self.run_and_return_best([weights])
        self.memory = [weights1, reward1]
 
        # Start loop
        for gen in range(self.generations + 1):

            # Mutate weights and make the mutations the new self.population
            self.mutate(weights)


            # Get best performing weights from the new self.population
            weights, reward = self.run_and_return_best(self.population)

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
            if gen % 10 == 0:
                print(f"Generation: {gen}")
                print(f"Best reward of generation: {reward:6f}")
                print(f"Terrain: Slope={env.slope:.2f}, Noise={env.noise:.2f}")
                print("")             

            # Save the best performing weights and the reward every n generations
            if gen % 10 == 0:
                self.best_per_gen.append([weights, reward])

        return self.best_per_gen

    
if __name__ == "__main__":

    # start = time.time()
    env = BipedalWalker()

    state_size, action_size = 24, 4
    network = NeuralNetwork(state_size, action_size)

    noise_range = [0.0, 1.1] # I want to include 1.0 
    slope_range = [-0.5, 0.5] # Slope 0.5 is already pretty steep

    step_size = 0.1
    terrain_params = generate_terrain(noise_range, slope_range, step_size) # 100 terrains at the moment with the current step size


    population_size = 10
    generations = 1
    xnes = EVO(population_size, env, network, terrain_params, generations)

    # Here goes nothing
    generalist_weights = xnes.main()
    
    # end = time.time()
    # print(f"Time taken: {end - start}")
    
    print(f"Final reward: {generalist_weights[-1][1]}")
    torch.save(generalist_weights[-1][0], "generalist-controllers-terrain/best_weights.pt")
