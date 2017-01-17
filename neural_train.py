# pylint: disable=C,R,no-member
# >>> neural_train.py allimages.npz output_directory

import tensorflow as tf
import numpy as np
from neural import CNN
from sys import argv
from time import time
import queue
import threading
import csv
import os

def main(training_images_path, training_labels_path, output_path):
    f = open(output_path + '/log.txt', 'w')
    fx = open(output_path + '/xent_batch.txt', 'w')

    cnn = CNN()

    f.write("{}\n".format(argv))
    f.flush()

    session = tf.Session()

    cnn.create_architecture()
    f.write("Architecture created\n")
    f.flush()

    saver = tf.train.Saver(max_to_keep=0)
    session.run(tf.global_variables_initializer())

    f.write("Session ready\n")
    f.flush()

    def print_log(xs, ys):
        ps = cnn.predict(session, xs)

        f.write('ys={}\nps={}\n'.format(ys, ps))

    def save_statistics(i):
        save_path = saver.save(session, '{}/{:05d}.data'.format(output_path, i))
        f.write('Model saved in file: {}\n'.format(save_path))


    # Use a Queue to generate batches and train in parallel
    n = 50000
    q = queue.Queue(100)  # max 100 batches in the queue

    def trainer():
        # Infinite loop waiting for batches to train the CNN
        for i in range(n+1):
            time_0 = time()

            xs, ys = q.get()

            time_qget = time()

            if i % 100 == 0 and i != 0:
                print_log(xs, ys)
                fx.flush()

            if i % 1000 == 0:
                xentropy = cnn.train_timeline(session, xs, ys, output_path + '/timeline{:05}.json'.format(i))
            else:
                xentropy = cnn.train(session, xs, ys)

            fx.write('{} {:.6}\n'.format(i, xentropy))

            if i % 1000 == 0 and i != 0:
                save_statistics(i)

            q.task_done()

            time_proc = time()
            f.write('{:05d}: {: >6.3f}s+{: >6.3f}s {} xent_batch={: >6.3f}\n'.format(
                i, time_qget - time_0, time_proc - time_qget,
                xs.shape, xentropy))
            f.flush()


    t = threading.Thread(target=trainer)
    t.daemon = True
    t.start()

    with open(training_labels_path) as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
        labels = np.array([r[1:] for r in rows[1:]])

    files = [training_images_path + '/' + f for f in sorted(os.listdir(training_images_path))]

    n_feeders = 4
    assert n % n_feeders == 0
    def feeder():
        f.write('feeder launched\n')
        f.flush()
        for _ in range(n // n_feeders):
            xs, ys = cnn.batch(files, labels)
            q.put((xs, ys))

    threads = [threading.Thread(target=feeder) for _ in range(n_feeders)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # the n+1
    xs, ys = cnn.batch(files, labels)
    q.put((xs, ys))

    q.join()
    session.close()

    f.close()
    fx.close()


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3])
