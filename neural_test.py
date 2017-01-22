# pylint: disable=C,R,no-member
# >>> neural_train.py arch.py allimages.npz output_directory

import tensorflow as tf
from sys import argv
from time import time, sleep
import queue
import threading
import os
import importlib.util
import sys

def predict_all(session, CNN, cnn, files, step=50):
    q = queue.Queue(20)  # batches in the queue

    def compute():
        while q.qsize() < 20:
            sleep(0.2)

        for _ in range(0, len(files), step):
            if q.qsize() < 5:
                sleep(0.1)
            ids, xs = q.get()

            ps = cnn.predict(session, xs)

            for i, p in zip(ids, ps):
                pstr = ",".join(["{:.4}".format(x) for x in p])
                print("{},{}".format(i, pstr))

            q.task_done()

    t = threading.Thread(target=compute)
    t.daemon = True
    t.start()

    for j in range(0, len(files), step):
        k = min(j + step, len(files))
        xs = CNN.load(files[j:k])
        ids = [f.split('/')[-1].split('.')[0] for f in files[j:k]]
        q.put((ids, xs))

    q.join()


def main(arch_path, images_path, restore_path):
    time_total_0 = time()

    spec = importlib.util.spec_from_file_location("module.name", arch_path)
    neural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural)
    CNN = neural.CNN

    cnn = CNN()

    print("GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6")

    session = tf.Session()

    cnn.create_architecture()

    saver = tf.train.Saver(max_to_keep=0)
    saver.restore(session, restore_path)

    files = [images_path + '/' + f for f in sorted(os.listdir(images_path))]

    predict_all(session, CNN, cnn, files)

    session.close()

    t = time() - time_total_0
    print("total time : {}h {}min".format(t // 3600, (t % 3600) // 60), file=sys.stderr)

if __name__ == '__main__':
    main(argv[1], argv[2], argv[3])
