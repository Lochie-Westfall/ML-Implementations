"""
Implementation of Proximal Policy Optimization by Lochie Westfall

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI

https://arxiv.org/pdf/1707.06347.pdf
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from ppo import ppo
import gym
import sys

# The name of the gym environment
# https://gym.openai.com/envs/
env_name = "Pendulum-v0"
episodes = 10000

# Create default session
sess = tf.Session().__enter__()

env = gym.make(env_name)
state = env.reset()

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

# Define shape of the value and policy networks
value_struct = [state_size, 100, 1]
policy_struct = [state_size, 100, action_size]

algorithm = ppo(value_struct, policy_struct)

states = []
rewards = []
actions = []

all_reward = []
moving_avg_reward = []

render_freq = 100
render_amt = 5

print_freq = 100

for episode in range(episodes):
    state = env.reset()
    done = False
    if episode>0:
        # Replace previous episode data print with current episode data 
        sys.stdout.write("\r" + "episode {} reward: {} \r".format(episode, all_reward[-1]))
        sys.stdout.flush()
    
    episode_reward = 0
    
    while not done:
      # Only render "render_amt" episodes for every "render_freq" episodes or when on macOS
      # Improves performance
      if episode % render_freq < render_amt or episode > episodes * 0.9 or sys.platform == "darwin":
          env.render()
      
      action = algorithm.get_action([state])
      state_p, reward, done, info = env.step(action)
      
      states.append(state)
      actions.append(action)
      rewards.append(reward) 
      state = state_p
      
      if done:
          all_reward.append(sum(rewards))  
          moving_avg_reward.append(sum(rewards)*0.1+moving_avg_reward[-1]*0.9 if episode > 0 else sum(rewards))
              
          rewards = algorithm.discount_and_normalize(rewards)
          # Train the ppo algorithm with the information in the buffers
          algorithm.train(np.vstack(states), np.vstack(actions), np.array(rewards)[:, np.newaxis])
          # Empty the buffers
          states = []
          rewards = []
          actions = []
      
    if episode > 0 and episode % print_freq == 0:
        reward_list = all_reward[-print_freq:-1]
        print("\nmean: {} std: {}".format(np.mean(reward_list), np.std(reward_list)))
        print("max: {} min: {}".format(np.max(reward_list), np.min(reward_list)))
        print(" ")

plt.plot(moving_avg_reward)
plt.title("{} trained by ppo".format(env_name))
plt.ylabel("Moving Averaged Reward")
plt.xlabel("Episode")
plt.show()