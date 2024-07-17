from XNES_BipedalWalker import generate_terrain
from biped_terrain import BipedalWalker
from network_old import NeuralNetwork, fill_parameters

import pickle as pkl
import keyboard as kb
import torch
import numpy as np



def test_good_bad_experient(generalists):
    """
    This function runs the complete network ensemble the BipedalWalker_split and BipedalWalker_morph
    experiments created.

    Params:
    data: Dictionary containing the generalists for each terrain

    """

    good = 0
    bad = 0

    noise_range = [0.0, 0.5]
    slope_range = [-0.4, 0.4]
    step_size = 0.1

    net = NeuralNetwork(24, 20, 4)

    terrain_params = generate_terrain(noise_range, slope_range, step_size)  # 40 terrains at the moment

    env = BipedalWalker(hardcore=True)

    fitnesses = []

    for params in terrain_params:
        env.noise, env.slope = params

        fits = []

        for generalist in generalists:
            fill_parameters(net, generalist)

            state, _ = env.reset()
            done = False
            score = 0
            s = 0

            while not done:
                action = net.forward(state).detach().numpy()

                state, reward, terminated, truncated, _ = env.step(action)

                score += reward     

                # env.render()

                # If the space bar is pressed, we skip to the next terrain simulation.
                if kb.is_pressed('space'):
                    break
                    
                # If the 'q' key is pressed, we close the environment and exit the program.
                if kb.is_pressed('q'):
                    env.close()
                    exit()
                
                if s > 1_600:
                    break

                s += 1
                
                done = terminated or truncated
            
            fits.append(score)

            env.close()

        
        fitnesses.append(max(fits))

    return fitnesses

if "__name__" == "__main__":    
    # import a pickle file
    with open('XNES_Biped\Experiment_Results\Run 1\generalists_dict_1.pkl', 'rb') as f:
        data = pkl.load(f)

    generalists = [torch.load(f"XNES_Biped\Experiment_Results\Run 1\generalist_ter_0_{i}.pt") for i in range(len(data))]
    scores = test_good_bad_experient(data, generalists)
    print("Scores:")
    print(scores)
    print()
    print(f"Average score: {np.mean(scores)}")
    print(f"Max score: {np.max(scores)}")