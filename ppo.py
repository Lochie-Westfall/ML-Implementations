#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:55:51 2018

@author: Lochie
"""

import tensorflow as tf
import networks

epsilon = 0.2
learning_rate = 1e-4

class PPO ():
    def __init__ (self, value_struct, policy_struct):
        self.state = tf.placeholder(tf.float32, [None, policy_struct[0]])
        self.action = tf.placeholder(tf.float32, [None, policy_struct[-1]])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        
        self.value_layers = networks.create_neural_net(self.state, value_struct)
        self.policy = networks.policy_network(self.state, policy_struct, "policy", True)
        self.old_policy = networks.policy_network(self.state, policy_struct, "old_policy", True)
        
        self.v = self.value_layers[-1]
        self.advantage = self.reward - self.v
        
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_train = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss)
        
        self.assign_old = [old_policy.assign(policy) for policy, old_policy in zip(self.policy.variables, self.old_policy.variables)]
        
        self.ratio = self.policy.distribution.prob(self.action) / self.old_policy.distribution.prob(self.action)
        
        self.actor_loss = -tf.reduce_mean(tf.minimum(self.advantage*self.ratio, self.advantage*tf.clip_by_value(self.ratio, 1-epsilon, 1+epsilon)))
        self.actor_train = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss)
        
        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())