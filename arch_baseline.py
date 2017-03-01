# pylint: disable=C,R,no-member
import math
import tensorflow as tf
import numpy as np
import layers_normal as nn


def dihedral(x, i):
    if len(x.shape) == 3:
        if i & 4:
            y = np.transpose(x, (1, 0, 2))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, ::-1]
        if i&3 == 2:
            return y[::-1, :]
        if i&3 == 3:
            return y[::-1, ::-1]

    if len(x.shape) == 4:
        if i & 4:
            y = np.transpose(x, (0, 2, 1, 3))
        else:
            y = x.copy()

        if i&3 == 0:
            return y
        if i&3 == 1:
            return y[:, :, ::-1]
        if i&3 == 2:
            return y[:, ::-1, :]
        if i&3 == 3:
            return y[:, ::-1, ::-1]

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
        x = nn.convolution(x, 16, w=6, s=2) # 210
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x) # 208
        summary_images(x, "layer2")
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)

        ########################################################################
        assert x.get_shape().as_list() == [None, 104, 104, 16]
        x = nn.convolution(x, 32) # 102
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x) # 100
        summary_images(x, "layer4")
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)

        ########################################################################
        assert x.get_shape().as_list() == [None, 50, 50, 32]
        x = nn.convolution(x, 64) # 48
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x) # 46
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x) # 44
        summary_images(x, "layer6")
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)

        ########################################################################
        assert x.get_shape().as_list() == [None, 22, 22, 64]
        x = nn.convolution(x, 128) # 20
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x) # 18
        x = nn.max_pool(x)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)

        ########################################################################
        assert x.get_shape().as_list() == [None, 9, 9, 128]
        x = nn.convolution(x, 256) # 7
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = nn.convolution(x, 1024, w=7)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 1024]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
        self.embedding_input = x

        x = nn.fullyconnected(x, 1024)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.fullyconnected(x, 1024)
        x = nn.batch_normalization(x, self.tfacc, with_gamma=True)
        x = tf.nn.dropout(x, self.tfkp)

        self.test = x
        x = nn.fullyconnected(x, 37, activation=None)

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
        tf.summary.image("input", x, 3)

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
        ids = np.random.choice(len(files), 20, replace=False)

        xs = CNN.load([files[i] for i in ids])
        ys = labels[ids]

        for i in range(len(xs)):
            s = np.random.uniform(0.8, 1.2)
            u = np.random.uniform(-0.1, 0.1)
            xs[i] = dihedral(xs[i], np.random.randint(8)) * s + u

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
