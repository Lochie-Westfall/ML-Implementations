#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:55:51 2018

@author: Lochie

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI
"""

import tensorflow as tf
import networks

gamma = 0.9
epsilon = 0.2
learning_rate = 1e-4
train_steps = 10

class PPO ():
    def __init__ (self, value_struct, policy_struct):
        # inputs
        self.state = tf.placeholder(tf.float32, [None, policy_struct[0]])
        self.action = tf.placeholder(tf.float32, [None, policy_struct[-1]])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        
        # Neural networks used in algorithm
        self.value_layers = networks.create_neural_net(self.state, value_struct)
        self.policy = networks.policy_network(self.state, policy_struct, "policy", True)
        self.old_policy = networks.policy_network(self.state, policy_struct, "old_policy", True)
        
        
        self.v = self.value_layers[-1]
        
        # (10) 
        self.advantage = self.reward - self.v
        
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_train = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss)
        
        # Chapter 3 line 1
        self.ratio = self.policy.distribution.prob(self.action) / self.old_policy.distribution.prob(self.action)
        
        # (7)
        self.actor_loss = -tf.reduce_mean(tf.minimum(self.advantage*self.ratio, self.advantage*tf.clip_by_value(self.ratio, 1-epsilon, 1+epsilon)))
        
        self.actor_train = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss)
        
        # Algorithm 1 line 7 " θold ← θ "
        self.assign_old = [old_policy.assign(policy) for policy, old_policy in zip(self.policy.variables, self.old_policy.variables)]
        
        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())
    
    def train (self, state, action, reward):
        self.sess.run(self.assign_old)
        # Algorithm 1 line 6 "Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT"
        for _ in train_steps:
            self.sess.run(self.actor_train, {self.state:state, self.action:action, self.reward:reward})
            self.sess.run(self.critic_train, {self.state:state, self.reward:reward})
    
    def get_action (self, state):
        return self.policy.get_action(state)
    
    def get_v (self, state):
        return self.sess.run(self.v, {self.state:state})
    
    def discount_rewards (rewards):
        discounted_rewards = []
        v = 0
        for r in reversed(rewards):
            v = r + gamma * v
            discounted_rewards.append(v)
        discounted_rewards.reverse()
        
        