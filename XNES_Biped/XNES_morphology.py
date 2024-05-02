
from biped_morphology import BipedalWalker
from network import NeuralNetwork, fill_parameters

import numpy as np
import time
import json
import pickle
import torch

from evotorch.algorithms import XNES
from evotorch.neuroevolution import NEProblem
from joblib import Parallel, delayed

def generate_morphologies(parameter1_range, parameter2_range, step_sizes):
    """
    The morphologies are generated in a way that the slope changes every generation,
    once all slopes are visited the noise is increased.

    This makes it so the task starts a bit easier in the beginning and it gets
    harder as the noise level increases.
    """

    parameter1_values = np.arange(parameter1_range[0], parameter1_range[1], step_sizes[0])
    parameter2_values = np.arange(parameter2_range[0], parameter2_range[1], step_sizes[1])

    morphologies = np.array(np.meshgrid(parameter1_values, parameter2_values)).T.reshape(-1, 2)

    return morphologies



class EVO():
    '''
    With the normal environment (noise=0.1, slope=0.0), the total_reward goal is 300,
    but since the varying morphologies make the environment harder, I will set the goal to 250.

    Parameters:
    - env: BipedalWalker environment
    - net: NeuralNetwork
    - morph_params: List of morphology parameters (noise, slope)
    - max_fitness: The target fitness the controller should achieve

    This version of the algorithm holds the reward of the agent for each morphology and split them into Good and Bad.
    Once the target reward is reached *for any morphology*, the evolution is continued in only the Bad morphology.
    '''

    def __init__(self, env : BipedalWalker, net : NeuralNetwork, morph_params: list, max_fitness = 250):
        self.env = env
        self.net = net
        self.morph_params = morph_params

        self.keep_morphs = morph_params.copy()

        self.max_fitness = max_fitness

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU !!!")
        else:
            print("Using CPU")
        self.net.to(self.device)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.morph_num  = 0

        self.generalists = []
        self.gen_morphs = []

        self.max_evals = 30 #115_000
        self.evals = 0

    def evaluation_function(self, net: NeuralNetwork):

        self.env.LEG_W = self.morph_params[self.morph_num][0] / 30.0
        self.env.LEG_H = self.morph_params[self.morph_num][1] / 30.0
        self.env.LEG_DOWN =  -(self.env.LEG_W) / 30.0

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

    def run(self, generations = 1000, pSize = 25, sigma = 0.1):

        print("Running EVO")

        problem = NEProblem(
            objective_sense="max",
            network_eval_func=self.evaluation_function,
            network=NeuralNetwork(self.state_size, 20, self.action_size),
            num_actors= 0, # Number of parallel evaluations.
            initial_bounds=(-0.00001, 0.00001)
                )

        searcher = XNES(
            problem,
            stdev_init = sigma,
            popsize = pSize
                )
        
        print("")
        print(f"Population: {searcher._popsize}, Generations: {generations}, Max_evals: {self.max_evals}")
        print("")

        count_split = 0

        for gen in range(generations):
            searcher.step()
            fitness = searcher.status["best"].evals[0].item()

            print(f"Generation: {gen}, Best Fitness: {fitness:.3f}")

            # Replace if statement with modulo calculation
            self.morph_num = (self.morph_num + 1) % len(self.morph_params) 

            if fitness >= self.max_fitness and self.morph_num == 0:
                
                good, bad, reached_goal, avg_fitness = self.split(searcher.status["best"].values)

                # This is a backup in case the code fails later on
                # save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/XNES_BipedWalker_ens_{count_split}.pt"
                # torch.save(searcher.status["best"].values, save_path)
                # count_split += 1

                self.morph_params = bad.copy()  # make the morphologies only the bad ones + 1 good for generalization
                self.gen_morphs.append(good)
                self.generalists.append(searcher.status["best"].values)

                net = NeuralNetwork(24, 20, 4)
                best_individual = searcher.status["best"].values
                fill_parameters(net, best_individual)

                new_problem = NEProblem(
                    objective_sense="max",
                    network_eval_func=self.evaluation_function,
                    network=net, # Hopefully the initial weights are the same as the best individual
                    num_actors= 0, # Number of parallel evaluations.
                    initial_bounds=(-0.00001, 0.00001)
                        )

                # I will  initialize a new searcher for the new set of morphologies.
                searcher = XNES(
                    new_problem,
                    stdev_init = sigma,
                    popsize = pSize
                        )

                
                if len(bad) == 0:
                    print("No morphologies left, stopping the evolution.")
                    print(f"Generation: {gen}, Final Best Fitness: {fitness:.3f}, Avg Fitness: {avg_fitness:.3f}")
                    break

            if self.evals >= self.max_evals:
                print("Max evaluations reached.\n   Exiting...\n")
                break
            
        if "bad" in locals():
            if len(bad) > 0:
                print(f"Morphologies that could not be solved: {bad}")

        self.merge_generalists(self.generalists)

        return searcher, self.generalists

    def split(self, best):
        """
        Test the best individual on all morphologies and split them into Good and Bad morphologies.
        Then continue the evolution on the Bad morphologies to encourage generalization.

        If the best individual reaches the target fitness on 85% of the morphologies, the initial goal is reached,
        but the evolution continues to hopefully find a better generalist controller.
        """

        # Fill in the parameters of the best individual
        fill_parameters(self.net, best)

        max_steps = 1600
        s = 0

        reached_goal = False

        good, bad, avg_score = [], [], []
        for param in self.morph_params:

            self.env.envs = param

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

            if score >= self.max_fitness: # Be a bit more lenient with accepting good and bad morphologies.
                good.append(param)
            else:
                bad.append(param)

            avg_score.append(score)

        if len(good) > int(len(self.morph_params) * 0.85):
          reached_goal = True


        self.env.close()

        print(f"  Evaluation: Good morphologies: {len(good)}, Bad morphologies : {len(bad)}")

        return good, bad, reached_goal, np.mean(avg_score)
    
    def merge_generalists(self, generalists) -> None:
        """
        Evaluate the generalists on all morphologies and select the best generalist controller
        for each morphology.

        Params:
        - generalists: List of generalist controllers
        - gen_terrians: List of morphologies for each generalist controller
        """

        if len(generalists) == 0:
            print("No generalists to merge, stopping the process.")
            
        
        elif len(generalists) == 1:
            print("Only one generalist, merging not possible, saving it to file.")
            filepath = "generalist-controllers-terrain\XNES_Biped\Experiment_Results\Generalists\generalists_dict.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump({0: self.gen_morphs[0]}, f)

        else:

            # Create a matrix with all fitnesses for each generalist on each morphology
            gen_matrix = []
            
            for _, generalist in enumerate(generalists):
                fill_parameters(self.net, generalist)
                
                controller_fit = []
                self.morph_params = self.keep_morphs.copy()
                for num in range(len(self.keep_morphs)):

                    self.morph_num = num
                    controller_fit.append(self.evaluation_function(self.net))

                gen_matrix.append(controller_fit)

            # Select the best generalist controller for each morphology, with no overlap
            generalists_new = {i: [] for i in range(len(generalists))}
            # ter_indeces = [i for i in range(len(gen_terrains[0]))]

            gen_matrix_T = np.array(gen_matrix).T
            for ter_idx, fits in enumerate(gen_matrix_T):

                best_fit = max(list(fits))
                contr = list(fits).index(best_fit)
                generalists_new[contr].append(self.keep_morphs[ter_idx])
            

            filepath = "XNES_Biped\Experiment_Results\Generalists\generalist_morph_dict.pkl"

            # Save the dictionary to a file
            with open(filepath, 'wb') as f:
                pickle.dump(generalists_new, f)

            print("Merging complete, generalists saved to file.")


def experiment():

    # Load the json file biped_exp.json
    with open('XNES_Biped/biped_exp.json') as file:
        data = json.load(file)

    start = time.time()

    env = BipedalWalker(envs=[8, 34]) # Initialize the environment with the default morphology

    state_size, hidden_size, action_size = data["NN-struc"]
    net = NeuralNetwork(state_size, hidden_size, action_size)

    leg_width = [5, 13] 
    leg_length = [26, 41] 
    step_size = [1, 2]

    morph_params = generate_morphologies(leg_width, leg_length, step_size) # 64 morphologies with the current bounds and step size

    sigma = data["stdev_init"]
    pSize = data["population"]  # At the moment the population is set manually at 30, but can be set chosen automatically by XNES (23)
    generations = data["generations"]
    target_fitness = data["targetFitness"]

    evo = EVO(env, net, morph_params, target_fitness)
    searcher, generalists = evo.run(generations = generations, pSize = pSize, sigma = sigma)

    end = time.time()
    print(f"Time taken: {(end - start) / 60} minutes") # Convert time to minutes and print it.

    # save_path = f"generalist-controllers-terrain/XNES_Biped/Experiment_Results/{data['filename']}_morph.pt"
    # torch.save(searcher.status["best"].values, save_path)

    return searcher, generalists

if __name__ == "__main__":
    searcher, generalists = experiment()

    # print(searcher.status["best"].values, searcher.status["best"].evals)

    # if generalists != []:
    for i, generalist in enumerate(generalists):
        torch.save(generalist, f"XNES_Biped/Experiment_Results/Generalists/generalist_morph_0_{i}.pt")
