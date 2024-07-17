from XNES_BipedalWalker_Split import generate_terrain
from biped_terrain import BipedalWalker
from network_old import NeuralNetwork, fill_parameters


import keyboard as kb
import torch
import numpy as np

def run_visualisation(net):
    """
    This function runs the visualisation of the BipedalWalker environment 
    for every terrain used in the experiments.
    """

    terrain_params = generate_terrain(noise_range=[0.0, 0.5], slope_range=[-0.4, 0.4], step_size=0.1) # 100 terrains at the moment with the current step size

    max_steps = 1600
    s = 0

    good = []
    bad = []
    scores = []

    env = BipedalWalker()

    for i, param in enumerate(terrain_params):
            
        env.noise, env.slope = param

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
        
        scores.append(score)
        # param1, param2 = param
        # print(f"Episode: [{param1:.1f}, {param2:.1f}] -  Score: {score}")
        if score >= 220:
            good.append(param)
        else:
            bad.append(param)

    env.close()

    return good, bad, scores

if "__name__" == "__main__":

    device = torch.device("cpu")

    load_path = "XNES_Biped/Experiment_Results/Generalists/generalist_ter_exp_3_1.pt"
    # load_path = "XNES_Biped/Experiment_Results/Results_Biped.pt"
    weights = torch.load(load_path)

    net = NeuralNetwork(24, 20, 4).eval()
    net.to(device)
    fill_parameters(net, weights)

    good, bad = run_visualisation(net)
    print(f"Good: {len(good)}, Bad: {len(bad)}")

    print("Closing Visualisation...")

