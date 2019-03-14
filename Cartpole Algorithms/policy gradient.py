import gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def create_neural_network(net_input, structure, name):
    with tf.variable_scope(name):
        out = net_input
        for i in range(1, len(structure)):
            out = tf.layers.dense(out, structure[i][0], structure[i][1])
        
        return out 

env = gym.make('CartPole-v0')

learning_rate_policy = 0.01
learning_rate_value = 0.1

policy_network_shape = (4, (5, None), (2, None)) # TODO:: automaticall find state and action size
value_network_shape = (4, (10, None), (1, None))

state_input = tf.placeholder(tf.float32, [None, policy_network_shape[0]], "state")
reward_input = tf.placeholder(tf.float32, [None, 1], "reward")
action_input = tf.placeholder(tf.float32, [None, 1], "action") # remove hardcoded action size
value_net = create_neural_network(state_input, value_network_shape, "value_net")
policy_net = create_neural_network(state_input, policy_network_shape, "policy_net")

dist = tf.distributions.Categorical(policy_net[0])        
sample = dist.sample(1)

advantage = reward_input - value_net[0]

value_loss = tf.reduce_mean(tf.square(advantage))
value_optimizer = tf.train.AdamOptimizer(learning_rate_value).minimize(value_loss) 

policy_loss = -tf.reduce_mean(advantage * tf.log(dist.prob(action_input)))
policy_optimizer = tf.train.AdamOptimizer(learning_rate_policy).minimize(policy_loss) 

sess = tf.Session()
sess.run(tf.initializers.global_variables())

best_reward = 0
episodes = 5000
record_freq = 10
gamma = 0.99

all_rewards = []
for episode in tqdm(range(episodes)):
    states_buffer = []
    rewards_buffer = []
    actions_buffer = []

    episode_reward = 0

    done = False
    state = env.reset()

    while not done:
#       if episode % 100 < 5:
#           env.render()
         
        output = sess.run(policy_net, {state_input: [state]})
        action = sess.run(sample, {state_input: [state]})
        state, reward, done, info = env.step(action[0])
        episode_reward += reward
       
        rewards_buffer.append(reward)
        states_buffer.append(state)
        actions_buffer.append(action)

    if episode % record_freq == 0: all_rewards.append(episode_reward)
    discounted_rewards = []
    discounted_reward = rewards_buffer[-1] 

    for reward in reversed(rewards_buffer):
        discounted_reward = reward + gamma * discounted_reward
        discounted_rewards.append([discounted_reward])
    sess.run(value_optimizer, {state_input: states_buffer, reward_input: discounted_rewards}) 
    sess.run(policy_optimizer, {state_input: states_buffer, action_input: actions_buffer, reward_input:discounted_rewards})

    if episode_reward >= best_reward * 0.75:
        best_reward = episode_reward
#       print(episode, ": ", episode_reward)

moving_averaged_reward = [all_rewards[0]]
for i in range(1,len(all_rewards)):
    moving_averaged_reward.append(all_rewards[i] * 0.1 + moving_averaged_reward[-1] * 0.9)

plt.plot([i * record_freq for i in range(len(all_rewards))], all_rewards)
plt.ylabel('reward') 
plt.show()