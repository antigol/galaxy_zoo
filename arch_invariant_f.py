# pylint: disable=C,R,no-member
import math
import tensorflow as tf
import numpy as np
import layers_dihedral_equi as nn


def summary_images(x, name):
    for i in range(min(4, x.get_shape().as_list()[3])):
        tf.summary.image("{}-{}".format(name, i), x[:, :, :, i:i+1])

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfy = None
        self.tfp = None
        self.mse = None
        self.tftrain_step = None
        self.tfkp = None
        self.tfacc = None
        self.tflr = None
        self.train_counter = 0
        self.test = None
        self.embedding_input = None


    def NN(self, x):
        assert x.get_shape().as_list() == [None, 424, 424, 3]
        xi, xr = nn.convolution2(x, None, 3, 1, w=4, s=2) # 211
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi, xr = nn.convolution2(xi, xr) # 209

        ########################################################################
        assert xi.get_shape().as_list() == [None, 209, 209, 3]
        assert xr.get_shape().as_list() == [None, 209, 209, 8*1]
        xi, xr = nn.convolution2(xi, xr, 6, 2, w=5, s=2) # 103
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi, xr = nn.convolution2(xi, xr) # 101

        ########################################################################
        assert xi.get_shape().as_list() == [None, 101, 101, 6]
        assert xr.get_shape().as_list() == [None, 101, 101, 8*2]
        xi, xr = nn.convolution2(xi, xr, 12, 4, w=5, s=2) # 49
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi, xr = nn.convolution2(xi, xr) # 47

        ########################################################################
        assert xi.get_shape().as_list() == [None, 47, 47, 12]
        assert xr.get_shape().as_list() == [None, 47, 47, 8*4]
        xi, xr = nn.convolution2(xi, xr, 24, 8, w=5, s=2) # 22
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi, xr = nn.convolution2(xi, xr) # 20

        ########################################################################
        assert xi.get_shape().as_list() == [None, 20, 20, 24]
        assert xr.get_shape().as_list() == [None, 20, 20, 8*8]
        xi, xr = nn.convolution2(xi, xr, 48, 16, w=4, s=2) # 9
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)

        ########################################################################
        assert xi.get_shape().as_list() == [None, 9, 9, 48]
        assert xr.get_shape().as_list() == [None, 9, 9, 8*16]
        xi, xr = nn.convolution2(xi, xr, 96, 32) # 7
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi, xr = nn.convolution2(xi, xr, w=7) # 1
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi = tf.nn.dropout(xi, self.tfkp)
        xr = tf.nn.dropout(xr, self.tfkp)

        ########################################################################
        assert xi.get_shape().as_list() == [None, 1, 1, 96]
        assert xr.get_shape().as_list() == [None, 1, 1, 8*32]
        xi = tf.reshape(xi, [-1, xi.get_shape().as_list()[-1]])
        xr = tf.reshape(xr, [-1, xr.get_shape().as_list()[-1]])
        self.embedding_input = xi

        xi, xr = nn.fullyconnected2(xi, xr, 192, 64)
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)
        xi = tf.nn.dropout(xi, self.tfkp)
        xr = tf.nn.dropout(xr, self.tfkp)

        xi, xr = nn.fullyconnected2(xi, xr)
        xi = nn.batch_normalization(xi, self.tfacc, input_repr='invariant')
        xr = nn.batch_normalization(xr, self.tfacc)

        self.test = xr
        with tf.name_scope('final_fc'):
            x = nn.fc(xi, 37, input_repr='invariant', output_repr='invariant')
            x += nn.fc(xr, 37, input_repr='regular', output_repr='invariant')
            x += tf.Variable(tf.constant(0.0, shape=[37]), name="b")

        ########################################################################
        assert x.get_shape().as_list() == [None, 37]

        c1 = tf.nn.softmax(x[:, 0:3])
        c2 = tf.nn.softmax(x[:, 3:5]) * c1[:, 1:2]
        c3 = tf.nn.softmax(x[:, 5:7]) * c2[:, 1:2]
        c4 = tf.nn.softmax(x[:, 7:9]) * c2[:, 1:2]
        c5 = tf.nn.softmax(x[:, 9:13]) * c2[:, 1:2]
        c6 = tf.nn.softmax(x[:, 13:15])
        c7 = tf.nn.softmax(x[:, 15:18]) * c1[:, 0:1]
        c8 = tf.nn.softmax(x[:, 18:25]) * c6[:, 0:1]
        c9 = tf.nn.softmax(x[:, 25:28]) * c2[:, 0:1]
        c10 = tf.nn.softmax(x[:, 28:31]) * c4[:, 0:1]
        c11 = tf.nn.softmax(x[:, 31:37]) * c4[:, 0:1]

        return tf.concat([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11], 1)


    def create_architecture(self):
        self.tfkp = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [], name="kp")
        self.tfacc = tf.placeholder_with_default(tf.constant(0.0, tf.float32), [], name="acc")
        self.tflr = tf.placeholder_with_default(tf.constant(1e-4, tf.float32), [], name="lr")

        x = self.tfx = tf.placeholder(tf.float32, [None, 424, 424, 3], name="input")
        tf.summary.image("input", x, 3)

        with tf.name_scope("nn"):
            self.tfp = self.NN(x)

        with tf.name_scope("cost"):
            self.tfy = tf.placeholder(tf.float32, [None, 37])
            self.mse = tf.reduce_mean(tf.square(self.tfp - self.tfy))

        with tf.name_scope("train"):
            self.tftrain_step = tf.train.AdamOptimizer(self.tflr).minimize(self.mse)

    @staticmethod
    def split_test_train(images_path, labels_csv):
        import csv
        import os
        with open(labels_csv) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            labels = np.array([[float(x) for x in r[1:]] for r in rows[1:]]).astype(np.float32)

        files = [images_path + '/' + f for f in sorted(os.listdir(images_path))]

        n = 3000 # for the test set
        return (files[:n], labels[:n]), (files[n:], labels[n:])

    @staticmethod
    def load(files):
        from scipy.ndimage import imread
        n = len(files)

        xs = np.zeros((n, 424, 424, 3), dtype=np.float32)
        for i in range(n):
            xs[i] = imread(files[i], mode='RGB').astype(np.float32) / 256.0

        return CNN.prepare(xs)

    @staticmethod
    def prepare(images):
        images = images - np.array([0.04543276, 0.04002843, 0.02984124])
        images = images / np.array([0.08930177, 0.0741211, 0.0656323])
        return images

    @staticmethod
    def batch(files, labels):
        ids = np.random.choice(len(files), 16, replace=False)

        xs = CNN.load([files[i] for i in ids])
        ys = labels[ids]

        for i in range(len(xs)):
            s = np.random.uniform(0.8, 1.2)
            u = np.random.uniform(-0.1, 0.1)
            xs[i] = xs[i] * s + u

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None, tensors=None):
        if tensors is None:
            tensors = []

        acc = 0.5 * 0.6 ** (self.train_counter / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (self.train_counter / 2000.0)
        lr = 1e-3 if self.train_counter < 50000 else 1e-4

        output = session.run([self.tftrain_step, self.mse] + tensors,
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc, self.tflr: lr},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return math.sqrt(output[1]), output[2:]

    def predict(self, session, xs):
        return session.run(self.tfp,
            feed_dict={self.tfx: xs})

    def predict_mse(self, session, xs, ys):
        return session.run([self.tfp, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys})
