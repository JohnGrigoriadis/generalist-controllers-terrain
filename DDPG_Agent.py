
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium as gym

import keyboard as kb

env = gym.make('BipedalWalker-v3', max_episode_steps=1600)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DDPG Agent class
class DDPGAgent:
    def __init__(self, state_size, action_size, load_path=None):

        # Using GPU will speed up the training process. So I hope it works.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.target_actor = Actor(state_size, action_size).to(self.device)
        self.target_critic = Critic(state_size, action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0002)

        self.update_target_models(tau=0.01)

        self.trained = False
        self.action_size = action_size

        # When reusing the model, the model can be loaded here.
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

    # def get_action(self, state, noise_std=0.7):
    #     state = torch.FloatTensor(state).to(self.device)
    #     with torch.no_grad():
    #         action = self.actor(state).cpu().numpy()
    #         action += noise_std * np.random.randn(self.action_size)
    #     return action
    
    # def get_action(self, state, noise_std=0.1):
    #     state = torch.FloatTensor(state).to(self.device)
    #     with torch.no_grad():
    #         action = self.actor(state).cpu().numpy()
    #         action += noise_std * np.random.randn(self.action_size)
    #     return action

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99):
        
        # Prints a message when the agent starts training
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
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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

        self.update_target_models(tau=0.01)


# Create the DDPG agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DDPGAgent(state_size, action_size, load_path=None)

def save_agent(model_name):
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()}, 
            f'Saved models/{model_name}')
    print(f"Model Saved in Saved models/{model_name}")
    
if __name__ == "__main__":
    print("The file is run directly")

    # Set the model name so it can be saved
    model_name = "ddpg_agent_test_3.pth"

    # action_list = []

    # Training the agent
    episodes = 100
    for epoch in range(episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        # Lists to store trajectory information
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while True:
            action = agent.get_action(state)
            # action_list.append(action)
            next_state, reward, done, truncated, _ = env.step(action)

            # Collect trajectory information
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

            if done or truncated:
                # Train the agent at the end of the episode
                agent.train(states, actions, rewards, next_states, dones)

                # Only print every n-th episode. Depends on the number of episodes
                if (epoch + 1) % 10 == 0:
                    print(f"Episode: {epoch + 1}, Total Reward: {total_reward}")
                break

            # In case the training shows no progress, the training can be stopped manually
            if kb.is_pressed("q"):
                env.close()
                print("Manual break")
                save_agent(model_name)
                exit()

        env.close()

    # Save the trained DDPG agent
    save_agent(model_name)

    # print(actions)
    
