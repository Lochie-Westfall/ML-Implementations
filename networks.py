#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:35:53 2018

@author: Lochie
"""

import tensorflow as tf
        
class policy_network ():
    def __init__(self, net_input, structure, name, trainable):
        self.name = name
        self.structure = structure
        self.trainable = trainable
        self.net_input = net_input
        
        self.create_network()
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.sess = tf.get_default_session()
        self.sess.run(tf.global_variables_initializer())
        
    def create_network (self):
        with tf.variable_scope(self.name):
            self.layers = create_neural_net(self.net_input, self.structure)
            
            loc = 2 * tf.layers.dense(self.layers[-1], self.structure[-1], tf.nn.tanh, trainable=self.trainable)
            scale = tf.layers.dense(self.layers[-1], self.structure[-1], tf.nn.softplus, trainable=self.trainable)
            self.distribution = tf.distributions.Normal(loc=loc, scale=scale)
        self.sample = tf.squeeze(self.distribution.sample(1),[0,1])
    
    def select_action (self, state):
        return  self.sess.run(self.sample, {self.layers[0]:state})
        
def create_neural_net (net_input, structure):
        layers = []
        layers.append(net_input)
        
        for i in range(1,len(structure)):
            layers.append(tf.layers.dense(layers[-1], structure[i]))
        
        return layers

    
    