# from XNES_BipedalWalker import generate_terrain
# from biped_terrain import BipedalWalker
from network import NeuralNetwork, fill_parameters

from biped_morphology import BipedalWalker
from XNES_morphology import generate_morphologies

import keyboard as kb
import torch
import numpy as np

# env = BipedalWalker()

def run_visualisation_morph(net):
    """
    This function runs the visualisation of the BipedalWalker environment 
    for every terrain used in the experiments.
    """

    noise_range = [5, 13] #[0.0, 0.7] 
    slope_range =  [26, 41]#[-0.5, 0.5] 
    step_size = [1, 2]#0.1

    terrain_params = generate_morphologies(noise_range, slope_range, step_size) # 100 terrains at the moment with the current step size

    max_steps = 1600
    s = 0

    good = 0
    bad = 0
    scores = []

    for i, param in enumerate(terrain_params):
        env = BipedalWalker(envs=param)
        # env.noise, env.slope = param

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
            
            if s > max_steps:
                break

            s += 1
            
            done = terminated or truncated

        # param1, param2 = param
        # print(f"Episode: [{param1:.1f}, {param2:.1f}] -  Score: {score}")
        if score > 220:
            good += 1
        else:
            bad += 1
        scores.append(score)

    env.close()
    return good, bad, scores

if __name__ == "__main__":
    
    device = torch.device("cpu")

    load_path = "XNES_Biped/Experiment_Results/Run 2/generalist_morph_1_2.pt"# "XNES_Biped\Experiment_Results\Results_Biped.pt"
    weights = torch.load(load_path)

    net = NeuralNetwork(24, 20, 4).eval()
    net.to(device)
    fill_parameters(net, weights)

    good, bad = run_visualisation_morph(net)
    print(f"Good: {good}, Bad: {bad}")
    print("Closing Visualisation...")