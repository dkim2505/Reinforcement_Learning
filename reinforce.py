import tensorflow as tf
import gym
import numpy as np

state = tf.placeholder(shape=[None, 4], dtype=tf.float32)
W=tf.Variable(tf.random_uniform([4,64], dtype=tf.float32))
hidden=tf.nn.relu(tf.matmul(state, W))
O=tf.Variable(tf.random_uniform([64,2], dtype=tf.float32))
output=tf.nn.softmax(tf.matmul(hidden,O))
rewards=tf.placeholder(shape=[None], dtype=tf.float32)
actions=tf.placeholder(shape=[None], dtype=tf.int32)
indices=tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs= tf.gather(tf.reshape(output, [-1]), indices)
loss =  -tf.reduce_mean(tf.log(actProbs)*rewards)
trainOp = tf.train.AdamOptimizer(.0001).minimize(loss)

gamma = 0.98
env = gym.make('CartPole-v1')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
history=[]
score=0
for i in range(10001):
    s = env.reset()
    for j in range(1000):
        act_prob = sess.run(output, feed_dict={state: [s]})
        act = np.random.choice(range(2), p=act_prob[0])
        next_s, r, dn, _ = env.step(act)
        history.append((s, act, r))
        score += r
        s = next_s
        if dn:
            R = 0
            for s, act, r in history[::-1]:
                R = r + gamma * R
                feed_dict = {state: [s], actions: [act], rewards: [R]}
                sess.run(trainOp, feed_dict)
            history=[]
            break
    if i%50 == 0:
        print("episode {} avg steps : {}".format(i, score/50))
        score = 0
