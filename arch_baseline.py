# pylint: disable=C,R,no-member
import tensorflow as tf
import numpy as np
import basic as nn

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

class CNN:
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.tfx = None
        self.tfp = None
        self.tfy = None
        self.tftrain_step = None
        self.mse = None
        self.tfkp = None
        self.tfacc = None
        self.train_counter = 0

        self.test = None

    def NN(self, x):
        assert x.get_shape().as_list() == [None, 424, 424, 3]
        x = nn.relu(nn.convolution(x, 8*4, w=4, s=2)) # 211
        x = nn.relu(nn.convolution(x)) # 209
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 209, 209, 8*4]
        x = nn.relu(nn.convolution(x, 8*8, w=5, s=2)) # 103
        x = nn.relu(nn.convolution(x)) # 101
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 101, 101, 8*8]
        x = nn.relu(nn.convolution(x, 8*16, w=5, s=2)) # 49
        x = nn.relu(nn.convolution(x)) # 47
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 47, 47, 8*16]
        x = nn.relu(nn.convolution(x, 8*32, w=5, s=2)) # 22
        x = nn.relu(nn.convolution(x)) # 20
        x = nn.batch_normalization(x, self.tfacc)

        ########################################################################
        assert x.get_shape().as_list() == [None, 20, 20, 8*32]
        x = nn.relu(nn.convolution(x, 8*64, w=4, s=2)) # 9
        x = nn.relu(nn.convolution(x)) # 7
        x = nn.relu(nn.convolution(x)) # 5
        x = nn.batch_normalization(x, self.tfacc)
        x = tf.nn.dropout(x, self.tfkp)

        ########################################################################
        assert x.get_shape().as_list() == [None, 5, 5, 8*64]
        x = nn.relu(nn.convolution(x, 8*128, w=5))

        ########################################################################
        assert x.get_shape().as_list() == [None, 1, 1, 8*128]
        x = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])

        x = tf.nn.dropout(x, self.tfkp)
        x = nn.relu(nn.fullyconnected(x, 8*256))
        x = tf.nn.dropout(x, self.tfkp)

        x = nn.relu(nn.fullyconnected(x, 8*256))
        x = nn.batch_normalization(x, self.tfacc)

        self.test = x
        x = nn.fullyconnected(x, 37)

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

        return tf.concat(1, [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11])


    def create_architecture(self):
        self.tfkp = tf.placeholder_with_default(tf.constant(1.0, tf.float32), [])
        self.tfacc = tf.placeholder_with_default(tf.constant(0.0, tf.float32), [])

        x = self.tfx = tf.placeholder(tf.float32, [None, 424, 424, 3])

        self.tfp = self.NN(x)

        self.tfy = tf.placeholder(tf.float32, [None, 37])
        self.mse = tf.reduce_mean(tf.square(self.tfp - self.tfy))

        self.tftrain_step = tf.train.AdamOptimizer(0.001, epsilon=1e-6).minimize(self.mse)

    @staticmethod
    def split_test_train(images_path, labels_csv):
        import csv
        import os
        with open(labels_csv) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            labels = np.array([[float(x) for x in r[1:]] for r in rows[1:]]).astype(np.float32)

        files = [images_path + '/' + f for f in sorted(os.listdir(images_path))]

        n = 2000 # for the test set
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

    def train(self, session, xs, ys, options=None, run_metadata=None):
        acc = 0.6 ** (self.train_counter / 1000.0)
        kp = 0.5 + 0.5 * 0.5 ** (self.train_counter / 2000.0)

        _, mse = session.run([self.tftrain_step, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: kp, self.tfacc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return mse

    def predict(self, session, xs):
        return session.run(self.tfp,
            feed_dict={self.tfx: xs, self.tfkp: 1.0, self.tfacc: 0.0})

    def predict_mse(self, session, xs, ys):
        return session.run([self.tfp, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 1.0, self.tfacc: 0.0})
