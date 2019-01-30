"""
Implementation of Proximal Policy Optimization by Lochie Westfall

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI

https://arxiv.org/pdf/1707.06347.pdf
"""

import tensorflow as tf
import numpy as np
import networks


# Discount factor
gamma = 0.9
# Clipping amount of actor loss
epsilon = 0.2
# Rate at which the policy and value networks update
learning_rate = 0.0001
# How many times to update each training
train_steps = 10

class ppo ():
    def __init__ (self, value_struct, policy_struct):
        self.state = tf.placeholder(tf.float32, [None, policy_struct[0]], name="state")
        self.action = tf.placeholder(tf.float32, [None, policy_struct[-1]], name="action")
        self.reward = tf.placeholder(tf.float32, [None, 1], name="reward")
        self.advantage_in = tf.placeholder(tf.float32, [None, 1], name="advantage_in")
        
        # Output of value network
        self.v = networks.create_neural_net(self.state, value_struct, True, "value")
        
        self.advantage = tf.subtract(self.reward, self.v, name="advantage")
        
        with tf.variable_scope("value_update"):
            # Update critic so that v is an estimator of average reward in given state
            self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
            self.critic_train = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss)
        
        self.policy = networks.policy_network(self.state, policy_struct, "policy", True)
        self.old_policy = networks.policy_network(self.state, policy_struct, "old_policy", False)
        
        # Probability ratio of policy and old policy
        self.ratio = tf.divide(self.policy.distribution.prob(self.action), self.old_policy.distribution.prob(self.action), name="ratio")
         
        with tf.variable_scope("policy_update"):
            # Calculate clipped surrogate loss
            self.actor_loss = -tf.reduce_mean(tf.minimum(self.advantage_in*self.ratio, self.advantage_in*tf.clip_by_value(self.ratio, 1-epsilon, 1+epsilon)))        
            self.actor_train = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss)
    
        with tf.variable_scope("assign_old"):
            # Assign current policy variables to old policy variables
            self.assign_old = [old_policy.assign(policy) for policy, old_policy in zip(self.policy.variables, self.old_policy.variables)]
            
        self.sess = tf.get_default_session()
        # Add all variables to tensorboard
        tf.summary.FileWriter("log/", self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

    def train (self, state, action, reward):
        self.sess.run(self.assign_old)
        
        for _ in range(train_steps):
            # Calculate advantage estimate for actor training
            # Necessary to stop actor update from affecting the critic network variables
            advantage = self.sess.run(self.advantage, {self.state:state, self.reward:reward})
            self.sess.run(self.actor_train, {self.state:state, self.action:action, self.advantage_in:advantage})
            self.sess.run(self.critic_train, {self.state:state, self.reward:reward})
    
    def get_action (self, state):
        return self.policy.get_action(state)
    
    def get_v (self, state):
        return self.sess.run(self.v, {self.state:state})[0,0]
    
    def discount_and_normalize (self, rewards):
        # Normalize the rewards for stability
        rewards = [(reward - np.mean(rewards))/abs(np.std(rewards)+1e-10) for reward in rewards]    
        
        discounted_rewards = []
            
        discounted_reward = rewards[-1]
        # Add all future rewards in the batch to the current reward by a decreasing amount 
        for reward in reversed(rewards):
            discounted_reward = reward + gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()

        return discounted_rewards
        
        