import tensorflow as tf
from sets import Mnist


class Convnet:

    def __init__(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.layer_convolution(x, 5, 32)
        x = self.layer_pooling(x)
        x = self.layer_convolution(x, 5, 64)
        x = self.layer_pooling(x)
        x = tf.reshape(x, [-1, 7 * 7 * 64])
        x = self.layer_dense(x, 1024)
        self.dropout = tf.placeholder(tf.float32)
        x = tf.nn.dropout(x, self.dropout)
        x = self.layer_softmax(x, 10)
        self.prediction = x

    def layer_convolution(self, x, width, depth):
        in_size = int(x.get_shape()[3])
        weight = self.create_weight([width, width, in_size, depth])
        bias = self.create_bias([depth])
        args = {'strides': [1, 1, 1, 1],
                'padding': 'SAME'}
        x = tf.nn.conv2d(x, weight, **args) + bias
        x = tf.nn.relu(x)
        return x

    def layer_pooling(self, x):
        args = {'ksize': [1, 2, 2, 1],
                'strides': [1, 2, 2, 1],
                'padding': 'SAME'}
        return tf.nn.max_pool(x, **args)

    def layer_dense(self, x, size):
        in_size = int(x.get_shape()[1])
        weight = self.create_weight([in_size, size])
        bias = self.create_bias([1024])
        x = tf.matmul(x, weight) + bias
        x = tf.nn.relu(x)
        return x

    def layer_softmax(self, x, size):
        in_size = int(x.get_shape()[1])
        weight = self.create_weight([in_size, size])
        bias = self.create_bias([size])
        x = tf.matmul(x, weight) + bias
        x = tf.nn.softmax(x)
        return x

    @staticmethod
    def create_weight(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def create_bias(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def optimization(y, pred):
    cross_entropy = -tf.reduce_sum(y * tf.log(pred))
    training = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    return training


def evaluation(y, pred):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


# Definition
x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])
model = Convnet(x)
training = optimization(y, model.prediction)
accuracy = evaluation(y, model.prediction)

# Session
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Training
train, test = Mnist()()
for i in range(20000):
    batch = train.random_batch(50)
    training.run(feed_dict={x: batch[0], y: batch[1], model.dropout: 0.5})
    if i % 100 == 0:
        testset = {x: test.data,
                   y: test.target,
                   model.dropout: 1.0}
        print('Step:', i, 'Test accuracy:', accuracy.eval(feed_dict=testset))
