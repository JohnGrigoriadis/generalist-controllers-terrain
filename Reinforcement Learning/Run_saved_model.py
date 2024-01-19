
import torch
import gymnasium as gym
from DDPG_Agent import  Actor, Critic, DDPGAgent # Import your actor module
import keyboard as kb
import time

env = gym.make("BipedalWalker-v3", render_mode="human", max_episode_steps=1000)

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Load the saved DDPG agent
agent = DDPGAgent(state_size, action_size, load_path='Saved moels/ddpg_agent.pth')

# Now, you can use loaded_agent for inference or further training

vis_epochs = 1

if input("Do you want to visualize the trained agent? (y/n): ") == 'y':
    # Assuming the 'agent' and 'env' are already defined and trained
    for epoch in range(vis_epochs):
        state, info = env.reset()
        total_reward = 0
        done = False

        start_time = time.time()

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, trunncated, _ = env.step(action)

            total_reward += reward

            env.render()

            if kb.is_pressed('space'):
                done = True

            if kb.is_pressed('q'):
                print("Manual break")
                env.close()
                exit()

            if trunncated:
                done = True

            state = next_state

        duration = time.time() - start_time
        print(f"Episode: {epoch + 1}, Total Reward: {total_reward}, Duration: {duration:.2f} seconds")

    env.close()

