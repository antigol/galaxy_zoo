# pylint: disable=C,R,no-member
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from math import sqrt
from scipy.ndimage import imread

def dihedral(src, i):
    dst = src
    if i & 4:
        dst = np.rot90(dst)
    if i & 1:
        dst = np.fliplr(dst)
    if i & 2:
        dst = np.flipud(dst)
    return dst

def convolution(x, f_out=None, s=1, w=3, padding='SAME', std=None):
    f_in = x.get_shape().as_list()[3]
    if f_out is None:
        f_out = f_in
    if std is None:
        std = sqrt(2.0 / (w * w * f_in))

    with tf.name_scope("conv_{}_{}".format(f_in, f_out)):
        W = tf.Variable(tf.truncated_normal([w, w, f_in, f_out], stddev=std), name="W")
        b = tf.Variable(tf.constant(0.1 * std, shape=[f_out]), name="b")
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding) + b

def pool22(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_normalization(x):
    depth = x.get_shape().as_list()[3]
    beta = tf.Variable(tf.constant(0.0, shape=[depth]))
    gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
    m, v = tf.nn.moments(x, axes=[0, 1, 2])
    return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)


class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfp = None
        self.tfy = None
        self.tftrain_step = None
        self.xentropy = None
        self.tfkp = None
        self.trainset = None

    def create_architecture(self):
        self.tfkp = tf.placeholder(tf.float32)

        x = self.tfx = tf.placeholder(tf.float32, [None, None, None, 3])

        x = tf.nn.relu(convolution(x, 16))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x)

        x = tf.nn.relu(convolution(x, 32))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x)

        x = tf.nn.relu(convolution(x, 64))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x)

        x = tf.nn.relu(convolution(x, 128))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x)

        x = tf.nn.relu(convolution(x, 256))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x)

        x = tf.nn.dropout(x, self.tfkp)
        x = tf.nn.relu(convolution(x, 512, w=1))
        x = tf.nn.dropout(x, self.tfkp)
        x = tf.nn.relu(convolution(x, 512, w=1))
        x = tf.nn.dropout(x, self.tfkp)
        x = convolution(x, 37, w=1)

        assert x.get_shape().as_list() == [None, None, None, 37]
        x = tf.reduce_sum(x, [1, 2])

        self.tfy = tf.placeholder(tf.float32, [None, 37])
        self.xentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(x, self.tfy))
        self.tfp = tf.nn.sigmoid(x)

        self.tftrain_step = tf.train.AdamOptimizer(0.001).minimize(self.xentropy)

    def batch(self, files, labels, n=10):
        ids = np.random.choice(len(files), n, replace=False)

        xs = np.zeros((n, 424, 424, 3))
        for i in range(n):
            xs[i] = imread(files[ids[i]], mode='RGB')
        ys = labels[ids]

        for i in range(len(xs)):
            xs[i] = dihedral(xs[i], np.random.randint(8))

        return xs, ys

    def train(self, session, xs, ys):
        _, xentropy = session.run([self.tftrain_step, self.xentropy],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 0.5})

        return xentropy

    def train_timeline(self, session, xs, ys, filename='timeline.json'):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        _, xentropy = session.run([self.tftrain_step, self.xentropy],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 0.5},
            options=run_options, run_metadata=run_metadata)

        # google chrome : chrome://tracing/

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(filename, 'w') as f:
            f.write(ctf)
        return xentropy

    def predict_naive(self, session, images):
        return session.run(self.tfp, feed_dict={self.tfx: images, self.tfkp: 1.0})

    def predict_naive_xentropy(self, session, images, labels):
        return session.run([self.tfp, self.xentropy], feed_dict={self.tfx: images, self.tfy: labels, self.tfkp: 1.0})

    def predict(self, session, images):
        # exploit symmetries to make better predictions
        ps = self.predict_naive(session, images)

        for i in range(1, 8):
            trans = np.copy(images)
            for j in range(len(trans)):
                trans[j] = dihedral(trans[j], i)
            ps *= self.predict_naive(session, trans)

        return ps

    def predict_xentropy(self, session, images, labels):
        # exploit symmetries to make better predictions
        ps, xent = self.predict_naive_xentropy(session, images, labels)

        for i in range(1, 8):
            trans = np.copy(images)
            for j in range(len(trans)):
                trans[j] = dihedral(trans[j], i)
            ps *= self.predict_naive(session, trans)

        return ps, xent
