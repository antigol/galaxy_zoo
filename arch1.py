# pylint: disable=C,R,no-member
import tensorflow as tf
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

def moments(x, axes):
    m = tf.reduce_mean(x, axes)
    v = tf.reduce_mean(tf.square(x), axes) - tf.square(m)
    return m, v

def batch_normalization(x, ub, acc):
    depth = x.get_shape().as_list()[3]

    with tf.name_scope("bn_{}".format(depth)):
        m, v = moments(x, axes=[0, 1, 2])

        acc_m = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False, name="acc_m")
        acc_v = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False, name="acc_v")

        new_acc_m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
        new_acc_v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)

        m = (1.0 - ub) * new_acc_m + ub * m
        v = (1.0 - ub) * new_acc_v + ub * v
        m.set_shape([depth])
        v.set_shape([depth])

        beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]))

        return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfp = None
        self.tfy = None
        self.tftrain_step = None
        self.xent = None
        self.tfkp = None
        self.ub = None
        self.acc = None
        self.train_counter = 0

    def create_architecture(self):
        self.tfkp = tf.placeholder(tf.float32)
        self.ub = tf.placeholder(tf.float32)
        self.acc = tf.placeholder(tf.float32)

        x = self.tfx = tf.placeholder(tf.float32, [None, None, None, 3])
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.relu(convolution(x, 16))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.relu(convolution(x, 32))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.relu(convolution(x, 64))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.relu(convolution(x, 128))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.relu(convolution(x, 256))
        x = tf.nn.relu(convolution(x))
        x = pool22(x)
        x = batch_normalization(x, self.ub, self.acc)

        x = tf.nn.dropout(x, self.tfkp)
        x = tf.nn.relu(convolution(x, 512, w=1))
        x = tf.nn.dropout(x, self.tfkp)
        x = tf.nn.relu(convolution(x, 512, w=1))
        x = tf.nn.dropout(x, self.tfkp)
        x = batch_normalization(x, self.ub, self.acc)

        x = convolution(x, 37, w=1)

        assert x.get_shape().as_list() == [None, None, None, 37]
        x = tf.reduce_sum(x, [1, 2])

        self.tfp = tf.nn.sigmoid(x)
        self.tfy = tf.placeholder(tf.float32, [None, 37])
        self.xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(x, self.tfy))

        self.tftrain_step = tf.train.AdamOptimizer(0.001).minimize(self.xent)

    @staticmethod
    def prepare(images_path, labels_csv):
        import csv
        import os
        with open(labels_csv) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            labels = np.array([[float(x) for x in r[1:]] for r in rows[1:]]).astype(np.float32)

        files = [images_path + '/' + f for f in sorted(os.listdir(images_path))]

        n = 10000 # for the test set
        return (files[:n], labels[:n]), (files[n:], labels[n:])

    @staticmethod
    def load(files):
        n = len(files)
        # 424x424
        xs = np.zeros((n, 424, 424, 3), dtype=np.float32)
        for i in range(n):
            xs[i] = imread(files[i], mode='RGB').astype(np.float32) / 256.0

        return xs

    @staticmethod
    def batch(files, labels, n=10):
        ids = np.random.choice(len(files), n, replace=False)

        xs = CNN.load([files[ids[i]] for i in range(n)])
        ys = labels[ids]

        for i in range(len(xs)):
            xs[i] = dihedral(xs[i], np.random.randint(8))

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None):
        ub = max(1.0 - self.train_counter / 10000, 0.0)
        acc = 0.0
        if self.train_counter < 10000:
            acc = 0.1
        elif self.train_counter < 20000:
            acc = 0.001

        _, xentropy = session.run([self.tftrain_step, self.xent],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 0.5, self.ub: ub, self.acc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return xentropy

    def train_timeline(self, session, xs, ys, filename='timeline.json'):
        from tensorflow.python.client import timeline
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        xentropy = self.train(session, xs, ys, run_options, run_metadata)
        # google chrome : chrome://tracing/

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(filename, 'w') as f:
            f.write(ctf)
        return xentropy

    def predict_xentropy(self, session, xs, ys):
        return session.run([self.tfp, self.xent],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 1.0, self.ub: 0.0, self.acc: 0.0})
