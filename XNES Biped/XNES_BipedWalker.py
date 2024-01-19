import gymnasium as gym
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evotorch.algorithms import XNES

env = gym.make("BipedalWalker-v3").unwrapped

maxGens = 10
maxFitness = 400 # The goal is 300 but pushing further isn't a bad thing
nStep = maxFitness
n_jobs = -1

inp = 24
hid = 32 # Lets play around with this to see the performance/complexity tradeoff
out = 4
popSize = 10
F1 = 0.1
CR = 0.2

class NeuralNetwork(nn.Module):
    def __init__(self, inp, hid, out):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(inp, hid)
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

net = NeuralNetwork(inp, hid, out)
optimizer = optim.Adam(net.parameters(), lr=0.01)

xnes = XNES(net.parameters(), popsize=popSize, num_interactions=nStep)

for gen in range(maxGens):
    solutions = xnes.ask(popSize)
    fitness = np.zeros(popSize)

    for i in range(popSize):
        net.load_state_dict(solutions[i])
        total_reward = 0

        for _ in range(nStep):
            observation, _ = env.reset()
            done = False

            while not done:
                action = net(torch.tensor(observation).float())
                observation, reward, done, tranculated, _ = env.step(action.detach().numpy())
                total_reward += reward

        fitness[i] = total_reward

    xnes.tell(fitness)
    best_solution = xnes.best_solution()
    best_fitness = fitness[np.argmax(fitness)]

    print(f"Generation: {gen+1}, Best Fitness: {best_fitness}")

