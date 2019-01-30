"""
Implementation of Proximal Policy Optimization by Lochie Westfall

Original Paper:
Proximal Policy Optimization Algorithms
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
OpenAI

https://arxiv.org/pdf/1707.06347.pdf
"""

import tensorflow as tf
        
class policy_network ():
    def __init__(self, net_input, structure, name, trainable):
        self.name = name
        self.structure = structure
        self.trainable = trainable
        self.net_input = net_input
    
        self.create_network()
        
        # Get variables of network
        # Needed to assign old policy variables 
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        self.sess = tf.get_default_session()
        
    def create_network (self):
        with tf.variable_scope(self.name):
            # Create policy network up to last hidden layer
            self.output = create_neural_net(self.net_input, self.structure[0:-1], self.trainable,self.name+"_net")
            
            # Create normal distribution as network output
            loc = 2 * tf.layers.dense(self.output, self.structure[-1], tf.nn.tanh, trainable=self.trainable)
            scale = tf.layers.dense(self.output, self.structure[-1], tf.nn.softplus, trainable=self.trainable)
            self.distribution = tf.distributions.Normal(loc=loc, scale=scale)
            # Operation used to select action
            self.sample = self.distribution.sample(1, name=self.name+"_sample")
    
    def get_action (self, state):
        return  self.sess.run(self.sample, {self.net_input:state})[0,0]
        
def create_neural_net (net_input, structure, trainable, name):
        with tf.variable_scope(name):
            layer = net_input
            
            # Loop through each number in network structure and create a layer of that size
            for i in range(1,len(structure)):
                layer = tf.layers.dense(layer, structure[i], tf.nn.relu, trainable=trainable)
            return layer

    
    