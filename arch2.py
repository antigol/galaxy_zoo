# pylint: disable=C,R,no-member
import tensorflow as tf
import numpy as np
import math

def dihedral_fullyconnected(x, f_out=None, std=None):
    f_in = x.get_shape().as_list()[1]
    assert f_in % 8 == 0
    if f_out is None:
        f_out = f_in
    assert f_out % 8 == 0
    if std is None:
        std = math.sqrt(2.0 / f_in)

    with tf.name_scope("fc_8x{}_8x{}".format(f_in // 8, f_out // 8)):
        ww = tf.Variable(tf.truncated_normal([f_in, f_out // 8], stddev=std), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[f_out // 8]), name="b")

        mt = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
            [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
            [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
            [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
        # tau[mt[a,b]] = tau[a] o tau[b]

        iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
        # tau[iv[a]] is the inverse of tau[a]

        wws = tf.split(0, 8, ww)

        W = tf.concat(1, [ # merge 8 part of the output
            tf.concat(0, [ # merge 8 part of the input
                wws[mt[iv[j], i]]
            for i in range(8)])
        for j in range(8)])

        return tf.matmul(x, W) + tf.tile(b, [8])


def dihedral_convolution(x, f_out=None, s=1, w=3, first=False, std=None, padding='SAME'):
    f_in = x.get_shape().as_list()[3]
    if f_out is None:
        f_out = f_in
    assert f_out % 8 == 0
    if std is None:
        std = math.sqrt(2.0 / (w * w * f_in))

    with tf.name_scope("conv_{}{}_8x{}".format('' if first else '8x', f_in if first else f_in//8, f_out//8)):
        ww = tf.Variable(tf.random_normal([w, w, f_in, f_out // 8], stddev=std), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[f_out // 8]), name="b")
        ws = [None] * 8
        ws[0] = ww  # tau[0]
        ws[1] = tf.reverse(ww, [False, True, False, False])  # tau[1]
        ws[2] = tf.reverse(ww, [True, False, False, False])  # tau[2]
        ws[3] = tf.reverse(ww, [True, True, False, False])  # tau[3]
        ws[4] = tf.transpose(ww, [1, 0, 2, 3])  # tau[4]
        ws[5] = tf.reverse(ws[4], [False, True, False, False])  # tau[5]
        ws[6] = tf.reverse(ws[4], [True, False, False, False])  # tau[6]
        ws[7] = tf.reverse(ws[4], [True, True, False, False])  # tau[7]
        # ws[j] = tau[j] F_all

        if first:
            W = tf.concat(3, ws)
        else:
            assert f_in % 8 == 0

            mt = np.array([
                [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
                [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
            # tau[mt[a,b]] = tau[a] o tau[b]

            iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
            # tau[iv[a]] is the inverse of tau[a]

            wws = [None] * 8
            for j in range(8):
                wws[j] = tf.split(2, 8, ws[j])
            # wws[j][i] = tau[j] F_i

            W = tf.concat(3, [ # merge 8 part of the output
                tf.concat(2, [ # merge 8 part of the input
                    wws[j][mt[iv[j], i]]
                for i in range(8)])
            for j in range(8)])

        # y = Conv(x, W)
        return tf.nn.conv2d(x, W, [1, s, s, 1], padding) + tf.tile(b, [8])

def dihedral_pool(x):
    shape = x.get_shape().as_list()
    f_in = shape[-1]
    assert f_in % 8 == 0

    with tf.name_scope("dihedral_pool_8x{}".format(f_in // 8)):
        xs = tf.split(len(shape) - 1, 8, x)
        return tf.div(tf.add_n(xs), 8.0)

def dihedral_batch_normalization(x, ub, acc):
    depth = x.get_shape().as_list()[3]
    assert depth % 8 == 0

    with tf.name_scope("bn_8x{}".format(depth // 8)):
        m, v = moments(dihedral_pool(x), axes=[0, 1, 2])

        acc_m = tf.Variable(tf.constant(0.0, shape=[depth // 8]), trainable=False, name="acc_m")
        acc_v = tf.Variable(tf.constant(0.0, shape=[depth // 8]), trainable=False, name="acc_v")

        new_acc_m = tf.assign(acc_m, (1.0 - acc) * acc_m + acc * m)
        new_acc_v = tf.assign(acc_v, (1.0 - acc) * acc_v + acc * v)

        m = tf.tile((1.0 - ub) * new_acc_m + ub * m, [8])
        v = tf.tile((1.0 - ub) * new_acc_v + ub * v, [8])
        m.set_shape([depth])
        v.set_shape([depth])

        beta = tf.tile(tf.Variable(tf.constant(0.0, shape=[depth // 8])), [8])
        gamma = tf.tile(tf.Variable(tf.constant(1.0, shape=[depth // 8])), [8])
        return tf.nn.batch_normalization(x, m, v, beta, gamma, 1e-3)

def convolution(x, f_out=None, s=1, w=3, padding='SAME', std=None):
    f_in = x.get_shape().as_list()[3]
    if f_out is None:
        f_out = f_in
    if std is None:
        std = math.sqrt(2.0 / (w * w * f_in))

    with tf.name_scope("conv_{}_{}".format(f_in, f_out)):
        W = tf.Variable(tf.truncated_normal([w, w, f_in, f_out], stddev=std), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[f_out]), name="b")
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
        self.tfl = None
        self.tfp = None
        self.tfy = None
        self.tftrain_step = None
        self.mse = None
        self.tfkp = None
        self.ub = None
        self.acc = None
        self.train_counter = 0

        self.test = None

    def create_architecture(self):
        self.tfkp = tf.placeholder(tf.float32)
        self.ub = tf.placeholder(tf.float32)
        self.acc = tf.placeholder(tf.float32)

        x = self.tfx = tf.placeholder(tf.float32, [None, 424, 424, 3])

        x = batch_normalization(x, self.ub, self.acc)

        def activation(x):
            return tf.nn.relu(x)

        x = activation(dihedral_convolution(x, 8 * 4, first=True))
        x = pool22(x)
        x = activation(dihedral_convolution(x))
        x = pool22(x)
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 106, 106, 8 * 4]

        x = activation(dihedral_convolution(x, 8 * 8))
        x = activation(dihedral_convolution(x, padding='VALID'))
        x = pool22(x)
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 52, 52, 8 * 8]

        x = activation(dihedral_convolution(x, 8 * 16))
        x = activation(dihedral_convolution(x))
        x = pool22(x)
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 26, 26, 8 * 16]

        x = activation(dihedral_convolution(x, 8 * 32))
        x = activation(dihedral_convolution(x, padding='VALID'))
        x = pool22(x)
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 12, 12, 8 * 32]

        self.test = x

        x = activation(dihedral_convolution(x, 8 * 64))
        x = activation(dihedral_convolution(x))
        x = pool22(x)
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 6, 6, 8 * 64]

        x = activation(dihedral_convolution(x, 8 * 128, padding='VALID'))
        x = activation(dihedral_convolution(x, w=4, padding='VALID'))
        x = dihedral_batch_normalization(x, self.ub, self.acc)
        assert x.get_shape().as_list() == [None, 1, 1, 8 * 128]
        x = tf.reshape(x, [-1, 8 * 128])

        x = dihedral_fullyconnected(x, 8 * 37)
        x = dihedral_pool(x)

        assert x.get_shape().as_list() == [None, 37]

        self.tfl = x
        self.tfp = tf.nn.sigmoid(x)
        self.tfy = tf.placeholder(tf.float32, [None, 37])
        self.mse = tf.reduce_mean(tf.square(self.tfp - self.tfy))

        self.tftrain_step = tf.train.AdamOptimizer(0.001).minimize(self.mse)

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
        from scipy.ndimage import imread
        n = len(files)

        xs = np.zeros((n, 424, 424, 3), dtype=np.float32)
        for i in range(n):
            xs[i] = imread(files[i], mode='RGB').astype(np.float32) / 256.0

        return xs

    @staticmethod
    def batch(files, labels):
        ids = np.random.choice(len(files), 10, replace=False)

        xs = CNN.load([files[i] for i in ids])
        ys = labels[ids]

        return xs, ys

    def train(self, session, xs, ys, options=None, run_metadata=None):
        ub = max(1.0 - self.train_counter / 10000, 0.0)
        acc = 0.0
        if self.train_counter < 10000:
            acc = 0.1
        elif self.train_counter < 20000:
            acc = 0.001

        _, mse = session.run([self.tftrain_step, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 0.5, self.ub: ub, self.acc: acc},
            options=options, run_metadata=run_metadata)

        self.train_counter += 1
        return mse

    def train_timeline(self, session, xs, ys, filename='timeline.json'):
        from tensorflow.python.client import timeline
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        mse = self.train(session, xs, ys, run_options, run_metadata)
        # google chrome : chrome://tracing/

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(filename, 'w') as f:
            f.write(ctf)
        return mse

    def predict(self, session, xs, ys):
        return session.run([self.tfp, self.mse],
            feed_dict={self.tfx: xs, self.tfy: ys, self.tfkp: 1.0, self.ub: 0.0, self.acc: 0.0})
