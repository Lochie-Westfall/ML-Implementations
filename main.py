"""
Implementation of Proximal Policy Optimization by Lochie Westfall

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI

The comments in the code reference the part of the paper they represent
"""

import ppo
import gym
import tensorflow as tf
import numpy as np

batch_size = 32
episodes = 10000
episode_length = 1000

sess = tf.Session().__enter__()

env = gym.make("Pendulum-v0")
state = env.reset()

action_size = env.action_space.sample()
action_size = len(action_size if (type(action_size)==list) else [action_size])

value_struct = [len(state), 100, 1]
policy_struct = [len(state), 100, action_size]

algorithm = ppo.ppo(value_struct, policy_struct)

states = []
rewards = []
actions = []
episode_reward = 0

for episode in range(episodes):
    state = env.reset()
    i = 0
    done = False
    while not done:
      i += 1
      if episode > episodes/2:
          env.render()

      action = algorithm.get_action([state])
      #print(action)
      state_p, reward, done, info = env.step(action)
      
      episode_reward += reward
      states.append(state)
      rewards.append([(reward+8)/8])
      actions.append(action)
      
      state = state_p
      
      if done:
          rewards = algorithm.discount_rewards(rewards, [state])
          algorithm.train(np.vstack(states), np.vstack(actions), rewards)
          
          states = []
          rewards = []
          actions = []
    if episode>0 and episode%1 == 0:
        print(episode, " ", episode_reward/1)
        episode_reward = 0