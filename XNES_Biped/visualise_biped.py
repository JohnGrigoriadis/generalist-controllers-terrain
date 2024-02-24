from XNES_BipedWalker import generate_terrain
from biped_terrain import BipedalWalker
from network import NeuralNetwork, fill_parameters, parameterize_net

import keyboard as kb
import torch
import numpy as np

noise_range = [0.0, 1.1] # I want to include 1.0 
slope_range = [-0.5, 0.6] # I want to include 1.0
step_size = 0.1

terrain_params = generate_terrain(noise_range, slope_range, step_size) # 100 terrains at the moment with the current step size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_path = "generalist-controllers-terrain\XNES_Biped\Results_Biped (2).pt"
weights = torch.load(load_path)

net = NeuralNetwork(24, 20, 4).eval()
net.to(device)
# net = parameterize_net(net, weights, device)
fill_parameters(net, weights)


env = BipedalWalker(render_mode="human")

def run_visualisation(env, net, terrain_params):
    """
    This function runs the visualisation of the BipedalWalker environment 
    for every terrain used in the experiments.
    """

    max_steps = 1600
    s = 0
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


        print(f"Episode: {param} Score: {score}, Terrain: {i}")

    env.close()
               
run_visualisation(env, net, terrain_params)
print(terrain_params[38])
print("Closing Visualisation...")