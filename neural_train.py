# pylint: disable=C,R,no-member
# >>> neural_train.py arch.py allimages.npz output_directory

import tensorflow as tf
import numpy as np
from sys import argv
from time import time, sleep
import queue
import threading
import importlib.util
from shutil import copy2
import os
import math

def predict_all(session, CNN, cnn, files, labels, f, step):
    q = queue.Queue(20)  # batches in the queue
    se_list = []

    def compute():
        for j in range(0, len(files), step):
            t0 = time()

            rem = len(files) // step - j // step
            if q.qsize() < min(2, rem):
                while q.qsize() < min(20, rem):
                    sleep(0.05)

            xs, ys = q.get()
            t1 = time()

            k = min(j + step, len(files))
            _, mse = cnn.predict_mse(session, xs, ys)
            se_list.append(mse * (k-j))

            t2 = time()
            f.write('{}/{} ({}) {: >6.3f}s+{:.3f}s\n'.format(
                j, len(files), q.qsize(), t1 - t0, t2 - t1))
            f.flush()

            q.task_done()

    t = threading.Thread(target=compute)
    t.daemon = True
    t.start()

    for j in range(0, len(files), step):
        k = min(j + step, len(files))
        xs = CNN.load(files[j:k])
        ys = labels[j:k]
        q.put((xs, ys))

    q.join()

    return np.sqrt(np.sum(se_list) / len(files))


def main(arch_path, images_path, labels_path, output_path, n_iter):
    time_total_0 = time()
    os.makedirs(output_path)
    os.makedirs(output_path + '/iter')
    f = open(output_path + '/log.txt', 'w')
    fm = open(output_path + '/metrics.txt', 'w')
    fx = open(output_path + '/rmse_batch.txt', 'w')

    fm.write("# iteration rmse_test rmse_train\n")
    fx.write("# iteration rmse_batch\n")

    copy2(arch_path, output_path + '/arch.py')
    spec = importlib.util.spec_from_file_location("module.name", arch_path)
    neural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural)
    CNN = neural.CNN

    cnn = CNN()

    f.write("{}\n".format(argv))
    f.flush()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    cnn.create_architecture()
    f.write("Architecture created\n")
    f.flush()

    saver = tf.train.Saver(max_to_keep=10)
    session.run(tf.global_variables_initializer())

    f.write("Session ready\n")
    f.flush()

    (test_files, test_labels), (train_files, train_labels) = CNN.split_test_train(images_path, labels_path)

    f.write("{: <6} images into train set\n".format(len(train_files)))
    f.write("{: <6} images into test set\n".format(len(test_files)))
    f.flush()

    def print_log(xs, ys):
        ps, mse = cnn.predict_mse(session, [xs[0]], [ys[0]])

        for c,i,k in [
            ("1:smooth/disk/star",0,3),
            ("2:edgeon/no",3,5),
            ("3:bar/no",5,7),
            ("4:spiral/no",7,9),
            ("5:bulge",9,13),
            ("6:odd:yes/no",13,15),
            ("7:rounded",15,18),
            ("8:ring/lens/...",18,25),
            ("9:bulge:rounded/boxy/no",25,28),
            ("10:spiral:tight/med/loose",28,31),
            ("11:spiral:1/2/3/4/+",31,37)]:
            text = " ".join(["{:.2f}/{:.2f}".format(p, y) for p,y in zip(ps[0][i:k], ys[0][i:k])])
            f.write("{} : {}\n".format(c, text))

        f.write("RMSE={}\n".format(math.sqrt(mse)))
        f.flush()

    def save_statistics(i):
        save_path = saver.save(session, '{}/iter/{:05d}.data'.format(output_path, i))
        f.write('Model saved in file: {}\n'.format(save_path))

        rmse_test = predict_all(session, CNN, cnn, test_files, test_labels, f, 25)
        rmse_train = predict_all(session, CNN, cnn, train_files[:len(test_files)], train_labels[:len(test_files)], f, 25)

        f.write("RMSE   test    train\n")
        f.write("     {: ^8.4} {: ^8.4}\n".format(rmse_test, rmse_train))
        f.flush()

        fm.write("{} {:.8g} {:.8g}\n".format(i, rmse_test, rmse_train))
        fm.flush()

    # Use a Queue to generate batches and train in parallel
    q = queue.Queue(50)  # batches in the queue

    def trainer():
        for i in range(n_iter+1):
            t0 = time()

            rem = n_iter + 1 - i
            if q.qsize() < min(3, rem):
                while q.qsize() < min(50, rem):
                    sleep(0.05)

            xs, ys = q.get()
            t1 = time()

            if i % 100 == 0 and i != 0 or i == 25:
                print_log(xs, ys)
                fx.flush()

            if i == 102 or i == 1002:
                from tensorflow.python.client import timeline
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                mse = cnn.train(session, xs, ys, options=run_options, run_metadata=run_metadata)
                # google chrome : chrome://tracing/

                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(output_path + '/timeline.json', 'w') as tlf:
                    tlf.write(ctf)
            else:
                mse = cnn.train(session, xs, ys)

            fx.write('{} {:.6}\n'.format(i, math.sqrt(mse)))

            if i % 1000 == 0 and i != 0 or i == 200:
                save_statistics(i)

            t2 = time()
            f.write('{:05d}: ({}) {: >6.3f}s+{:.3f}s {} RMSE_batch={:.3f} (MSE={:.3f})\n'.format(
                i, q.qsize(), t1 - t0, t2 - t1, xs.shape, math.sqrt(mse), mse))
            f.flush()

            q.task_done()

    t = threading.Thread(target=trainer)
    t.daemon = True
    t.start()

    f.write(" Done\nStart feeders...")
    f.flush()

    # the n+1
    xs, ys = CNN.batch(train_files, train_labels)
    q.put((xs, ys))

    n_feeders = 2
    assert n_iter % n_feeders == 0
    def feeder():
        for _ in range(n_iter // n_feeders):
            xs, ys = CNN.batch(train_files, train_labels)
            q.put((xs, ys))

    threads = [threading.Thread(target=feeder) for _ in range(n_feeders)]
    for t in threads:
        t.start()
    f.write("Done\n")
    f.flush()
    for t in threads:
        t.join()

    q.join()
    session.close()

    t = time() - time_total_0
    f.write("total time : {}h {}min".format(t // 3600, (t % 3600) // 60))

    f.close()
    fm.close()
    fx.close()


if __name__ == '__main__':
    main(argv[1], argv[2], argv[3], argv[4], int(argv[5]))
