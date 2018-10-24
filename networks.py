"""
Implementation of Proximal Policy Optimization by Lochie Westfall

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI

The comments in the code reference the part of the paper they represent
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
            self.layers = create_neural_net(self.net_input, self.structure, self.trainable)
            
            loc = 2 * tf.layers.dense(self.layers[-1], self.structure[-1], tf.nn.tanh, trainable=self.trainable)
            scale = tf.layers.dense(self.layers[-1], self.structure[-1], tf.nn.softplus, trainable=self.trainable)
            self.distribution = tf.distributions.Normal(loc=loc, scale=scale)
        self.sample = tf.squeeze(self.distribution.sample(1),[0,1])
    
    def get_action (self, state):
        #outputs nan
        return  self.sess.run(self.sample, {self.layers[0]:state})
        
def create_neural_net (net_input, structure, trainable=True):
        layers = []
        layers.append(net_input)
        
        for i in range(1,len(structure)):
            layers.append(tf.layers.dense(layers[-1], structure[i], tf.nn.relu, trainable=trainable))
        
        return layers

    
    