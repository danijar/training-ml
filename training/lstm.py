import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell
from sets import Mnist


class Lstm:

    def __init__(self, x, size):
        x, self.states = self.layer_lstm(x, size)
        x = self.layer_softmax(x, 10)
        self.prediction = x

    def layer_lstm(self, x, size):
        cell = BasicLSTMCell(size)
        outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
        # We only care about the LSTM outputs at the last timestep.
        x = outputs[-1]
        return x, states

    def layer_softmax(self, x, size):
        x = self.layer_linear(x, size)
        x = tf.nn.softmax(x)
        return x

    def layer_linear(self, x, size):
        in_size = int(x.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([in_size, size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[size]))
        x = tf.matmul(x, weight) + bias
        return x


def optimization(y, pred):
    cross_entropy = -tf.reduce_sum(y * tf.log(pred))
    training = tf.train.RMSPropOptimizer(0.005).minimize(cross_entropy)
    return training


def evaluation(y, pred):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def to_sequences(data, target, x, y):
    frames = []
    for i in range(28):
        frames.append(data[:, i])
    feed = {k: v for k, v in zip(x, frames)}
    feed[y] = target
    return feed


# Definition
seq_len = 28
x = [tf.placeholder(tf.float32, [None, 28]) for _ in range(seq_len)]
y = tf.placeholder(tf.float32, [None, 10])
model = Lstm(x, 200)
training = optimization(y, model.prediction)
accuracy = evaluation(y, model.prediction)

# Session
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Training
train, test = Mnist()()
test_feed = to_sequences(test.data, test.target, x, y)
for i in range(4000):
    data, target = train.random_batch(50)
    training.run(feed_dict=to_sequences(data, target, x, y))
    if i % 100 == 0:
        print('Step:', i, 'Test accuracy:', accuracy.eval(feed_dict=test_feed))
