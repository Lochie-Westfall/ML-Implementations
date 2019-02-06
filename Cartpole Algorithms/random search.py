import gym
import tensorflow as tf


def create_neural_network (net_input, structure, name):
    with tf.variable_scope(name):
        out = net_input
        for i in range(1, len(structure)):
            out = tf.layers.dense(out, structure[i][0], structure[i][1])

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return out, variables


env = gym.make('CartPole-v0')

network_shape = ((4), (1, tf.nn.relu))
print(network_shape[0])
input = tf.placeholder(tf.float32, [None, network_shape[0]])

_, best_neural_net_vars = create_neural_network(input, network_shape, "best_net")
neural_net, neural_net_vars = create_neural_network(input, network_shape, "test_net")

for var in neural_net_vars:
    print(var.value)

update_current_variables = [current.assign(best) for current, best in zip(neural_net_vars, best_neural_net_vars)]
update_best_variables = [best.assign(current) for current, best in zip(neural_net_vars, best_neural_net_vars)]

mutate_network_variables = [variable.assign(variable + (tf.random_uniform(tf.shape(variable))-0.5) * 0.01) for variable in neural_net_vars]
randomize_network_variables = [variable.assign(tf.random_uniform(tf.shape(variable))) for variable in neural_net_vars]

sess = tf.Session()

sess.run(tf.initializers.global_variables())

best_reward = 0

episodes = 100

for episode in range(episodes):

    if best_reward == 200:
        sess.run(update_current_variables)
    else:
        sess.run(randomize_network_variables)

    done = False
    state = env.reset()
    episode_reward = 0

    while not done:
        env.render()
        action = sess.run(neural_net, {input: [state]})[0][0]
        state, reward, done, info = env.step(int(round(action)))
        episode_reward += reward

    if episode_reward > best_reward:
        print("Reached", episode_reward, "score in episode", episode + 1)
        best_reward = episode_reward
        sess.run(update_best_variables)

