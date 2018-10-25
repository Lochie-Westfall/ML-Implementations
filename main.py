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
#import Box2D

from sys import platform

batch_size = 32
episodes = 100000
episode_length = 1000

sess = tf.Session().__enter__()

env = gym.make("Pendulum-v0")
state = env.reset()

action_size = env.action_space.sample()
action_size = len(action_size if (type(action_size)==np.ndarray) else [action_size])



value_struct = [len(state), 100, 1]
policy_struct = [len(state), 100, action_size]

algorithm = ppo.ppo(value_struct, policy_struct)

states = []
rewards = []
actions = []
episode_reward = 0

for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
      
      # Improves training speed
      if episode > episodes/2 or platform == "darwin":
          env.render()
          
      action = algorithm.get_action([state])

      state_p, reward, done, info = env.step(action)
      
      states.append(state)
      actions.append(action)
      rewards.append(reward)     
      
      state = state_p
      
      if done:
          episode_reward += sum(rewards)
          rewards = algorithm.discount_rewards(rewards, [state])
          
          algorithm.train(np.vstack(states), np.vstack(actions), np.array(rewards)[:, np.newaxis])
          
          states = []
          rewards = []
          actions = []
    if episode>0 and episode%1 == 0:
        print(episode, " ", episode_reward/1)
        episode_reward = 0