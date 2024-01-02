# In this example, I added manual override to stop the episode and close the environment.

import gymnasium as gym
import keyboard as kb

env = gym.make("BipedalWalker-v3", render_mode="human", max_episode_steps=1600)

episodes = 10

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        # For BipedalWalker, the action space is continuous, and we need to provide two values.
        # In this example, we choose random continuous actions between -1 and 1 for both actions.
        action = env.action_space.sample()

        _, reward, done, _, _ = env.step(action)
        score += reward     
        env.render()

        # If the space bar is pressed done is set to True and the episode ends.
        if kb.is_pressed('space'):
            done = True

        if kb.is_pressed('q'):
            env.close()
            break

    print(f"Episode: {episode} Score: {score}")

env.close()
