
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium as gym

import keyboard as kb

env = gym.make('BipedalWalker-v3', max_episode_steps=1000)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 4)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DDPG Agent class
class DDPGAgent:
    def __init__(self, state_size, action_size, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.target_actor = Actor(state_size, action_size).to(self.device)
        self.target_critic = Critic(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        self.update_target_models(tau=0.1)

        self.trained = False

        if load_path is not None:
            self.load_agent(load_path)

    def load_agent(self, load_path):
        checkpoint = torch.load(load_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


    def update_target_models(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return action

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99):
        
        if not self.trained: 
            print("Agent is training")
            self.trained=True 

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        target_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, target_actions)
        target_values = rewards + (1 - dones) * gamma * target_q_values

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = nn.functional.mse_loss(self.critic(states, actions), target_values)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_target_models(tau=0.1)

# Create the DDPG agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DDPGAgent(state_size, action_size, load_path=None)

if __name__ == "__main__":
    print("This is a module, not a script. Please import this module to use it.")
    # Training the agent
    episodes = 20
    for epoch in range(episodes):
        state, info = env.reset()
        total_reward = 0

        while True:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.train([state], [action], [reward], [next_state], [done])

            state = next_state
            total_reward += reward

            if done:
                print("Episode: {}, Total Reward: {}".format(epoch + 1, total_reward))
                break

            if truncated:
                print("Episode: {}, Total Reward: {}".format(epoch + 1, total_reward))
                break

    env.close()

    # Save the trained DDPG agent
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()}, 
            'Getting somewhere/ddpg_agent.pth')

