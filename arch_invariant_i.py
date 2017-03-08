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
        self.train_counter = 0
        self.test = None
        self.embedding_input = None


    def NN(self, x):
        assert x.get_shape().as_list() == [None, 424, 424, 3]
        summary_images(x, "layer0")
        x = nn.convolution(x, 8*4, w=4, s=2, input_repr='invariant') # 211
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x) # 209
        summary_images(x, "layer2")

        ########################################################################
        assert x.get_shape().as_list() == [None, 209, 209, 8*4]
        x = nn.convolution(x, 8*8, w=5, s=2) # 103
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x) # 101
        summary_images(x, "layer4")

        ########################################################################
        assert x.get_shape().as_list() == [None, 101, 101, 8*8]
        x = nn.convolution(x, 8*16, w=5, s=2) # 49
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x) # 47
        summary_images(x, "layer6")

        ########################################################################
        assert x.get_shape().as_list() == [None, 47, 47, 8*16]
        x = nn.convolution(x, 8*32, w=5, s=2) # 22
        x = nn.batch_normalization(x, self.tfacc)
        x = nn.convolution(x) # 20

        ########################################################################
        assert x.get_shape().as_list() == [None, 20, 20, 8*32]
        x = nn.convolution(x, 8*64, w=4, s=2) # 9
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 9, 9, 8*64]
        x = nn.convolution(x, 256, output_repr='invariant', activation=None) # 7
        x = nn.convolution(x, 8*256, w=7, input_repr='invariant')
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 8*256]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        self.embedding_input = x

        x = nn.fullyconnected(x, 8*256)
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 8*256)
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        self.test = x
        x = nn.fullyconnected(x, 37, activation=None, output_repr='invariant')

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

        x = self.tfx = tf.placeholder(tf.float32, [None, 424, 424, 3], name="input")

        with tf.name_scope("nn"):
            self.tfp = self.NN(x)

        with tf.name_scope("cost"):
            self.tfy = tf.placeholder(tf.float32, [None, 37])
            self.mse = tf.reduce_mean(tf.square(self.tfp - self.tfy))

        with tf.name_scope("train"):
            self.tftrain_step = tf.train.AdamOptimizer(1e-4).minimize(self.mse)

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
        ids = np.random.choice(len(files), 8, replace=False)

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

        acc = 0.6 ** (self.train_counter / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (self.train_counter / 2000.0)

        output = session.run([self.tftrain_step, self.mse] + tensors,
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return math.sqrt(output[1]), output[2:]

    def predict(self, session, xs):
        return session.run(self.tfp,
            feed_dict={self.tfx: xs})

    def predict_mse(self, session, xs, ys):
        return session.run([self.tfp, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys})
