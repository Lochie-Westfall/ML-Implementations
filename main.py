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

batch_size = 256
episodes = 1000
episode_length = 1000

sess = tf.Session().__enter__()

env = gym.make("Pendulum-v0")
observation = env.reset()

value_struct = [len(observation), 100, 1]
policy_struct = [len(observation), 100, len([env.action_space.sample()])]

algorithm = ppo.ppo(value_struct, policy_struct)

states = []
rewards = []
actions = []
episode_reward = 0

for episode in range(episodes):
    observation = env.reset()
    i = 0
    done = False
    while not done:
      i += 1
      env.render()

      action = algorithm.get_action([observation])[0]
      observation, reward, done, info = env.step(action)
      
      episode_reward += reward
      
      states.append(observation)
      rewards.append([reward])
      actions.append([action])
      
      if done or ((i % batch_size == 0 or i == episode_length-1)):
          rewards = algorithm.discount_rewards(rewards)
          algorithm.train(states, actions, rewards)
          
          states = []
          rewards = []
          actions = []
    if episode%1 == 0:
        print(episode, " ", episode_reward/1)
        episode_reward = 0