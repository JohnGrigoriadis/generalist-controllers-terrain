
from biped_terrain import BipedalWalker
# from network import NeuralNetwork

import numpy as np
import torch
import time
import json

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")

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
        self.terrain_params = terrain_params
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
        self.eval = 0
        self.good_terrains = []
        self.bad_terrains = []

        self.avg_terrain = terrain_params[60]

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
            network=NeuralNetwork(24, 20, 4).eval(),
            num_actors= 0, # Number of parallel evaluations.
            initial_bounds=(-0.00001, 0.00001)
                )

        searcher = XNES(
            problem,
            stdev_init = sigma,
            popsize = pSize
                )
        print(f"Population: {searcher._popsize}, Generations: {generations}")
        print("")

        save = True
        count = 0
        count_split = 0

        for gen in range(generations):
            searcher.step()
            fitness = searcher.status["best"].evals[0].item()


            if gen % 25 == 0:
                print(f"Generation: {gen}, Best Fitness: {fitness:.3f}")

            # Replace if statement with modulo calculation
            self.ter_num = (self.ter_num + 1) % len(self.terrain_params) 
            # self.ter_num += 1
            # if self.ter_num == len(self.terrain_params):
            #     self.ter_num =  0

            # Save the first individual that reaches a fitness of 250
            # More or less a fail-safe to keep at least one good individual, in case the evolution fails after a certain point.
            # Won't use it for now.
            # if fitness[0].item() >= self.max_fitness and save:
            #     save_path = "generalist-controllers-terrain/XNES_Biped/XNES_BipedWalker_250.pt"
            #     torch.save(searcher.status["best"].values, save_path)
            #     save = False


            if fitness >= self.max_fitness and self.ter_num == 0:
                good, bad, reached_goal, avg_fitness = self.split(searcher.status["best"].values)

                save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/XNES_BipedWalker_dif_{count_split}.pt"
                torch.save(searcher.status["best"].values, save_path)
                count_split += 1

                self.terrain_params = bad + [self.avg_terrain] # make the terrains only the bad ones + 1 good for generalization

                self.ter_num = 0 # reset ter count no avoid out of range errors
                # self.good_terrains.extend(good) # not sure why i do this, but sure.

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

                # Cannot set the best fitness to -200, because the evals attribute is a readonly tensor
                # searcher.status["best"].evals = torch.tensor([-200.0]) # Set the best fitness to -200 so the next generation will be evaluated.

                if reached_goal:
                    save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/Goal_BipedWalker_{len(good)}.pt"
                    count += 1
                    torch.save(searcher.status["best"].values, save_path)
                    print(f"Generation: {gen}, Final Best Fitness: {fitness:.3f}, Avg Fitness: {avg_fitness:.3f}")
                
                if len(bad) == 0:
                    print("No bad terrains left, stopping the evolution.")
                    print(f"Generation: {gen}, Final Best Fitness: {fitness:.3f}, Avg Fitness: {avg_fitness:.3f}")
                    break

        return searcher

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

        terrain_params = generate_terrain([0.0, 1.1], [-0.5, 0.6], 0.1)

        good, bad, avg_score = [], [], []
        for param in terrain_params:

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

        self.eval += 1
        return good, bad, reached_goal, np.mean(avg_score)

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
    generations = data["generations"]
    target_fitness = data["targetFitness"]

    evo = EVO(env, net, terrain_params, target_fitness)
    searcher = evo.run(generations = generations, pSize = pSize, sigma = sigma)

    end = time.time()
    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/{data['filename']}.pt"
    torch.save(searcher.status["best"].values, save_path)

    return searcher

if __name__ == "__main__":
    searcher = experiment()

    print(searcher.status["best"].values, searcher.status["best"].evals)
