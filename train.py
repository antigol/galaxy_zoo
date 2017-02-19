# pylint: disable=C,R,no-member

# Usage
# python3 neural_train.py arch?/arch.py path_to_npz_files output_path number_of_iteration

import math
import tensorflow as tf
import numpy as np
from sys import argv
from time import time, sleep
import queue
import threading
import importlib.util
from shutil import copy2
import os
import sys
import glob
import scipy
import scipy.ndimage as ndimage

def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min_ = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min_).transpose(3,0,1,2)
    max_ = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max_).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def predict_all(session, CNN, cnn, files, labels, f, step=50):
    q = queue.Queue(20)  # batches in the queue
    se_list =  []

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
    if os.path.isdir(output_path):
        resume = True
        if not os.path.isdir(output_path + '/iter'):
            sys.exit("Try to resume computation : no iter dir in the directory")
        if not arch_path.startswith(output_path):
            sys.exit("Try to resume computation : you need to resume with the same architecture")
        f = open(output_path + '/log.txt', 'a')
        fm = open(output_path + '/metrics.txt', 'a')
        fx = open(output_path + '/rmse_batch.txt', 'a')
    else:
        resume = False
        os.makedirs(output_path)
        os.makedirs(output_path + '/iter')
        f = open(output_path + '/log.txt', 'w')
        fm = open(output_path + '/metrics.txt', 'w')
        fx = open(output_path + '/rmse_batch.txt', 'w')

        f.write("{}\n".format(argv))
        f.flush()

        copy2(arch_path, output_path + '/arch.py')

    f.write("Loading {}...".format(arch_path))
    f.flush()

    spec = importlib.util.spec_from_file_location("module.name", arch_path)
    neural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neural)
    CNN = neural.CNN

    cnn = CNN()

    f.write(" Done\nSplit data set...")
    f.flush()

    (files_test, labels_test), (files_train, labels_train) = CNN.split_test_train(images_path, labels_path)

    f.write(" Done\n")
    f.write("{: <6} images into train set\n".format(len(files_train)))
    f.write("{: <6} images into test set\n".format(len(files_test)))
    f.write("Create TF session...")
    f.flush()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    session = tf.Session(config=config)

    f.write(" Done\nCreate graph...")
    f.flush()

    cnn.create_architecture()
    f.write("Done\n")
    f.flush()

    f.write("Merge summary...")
    f.flush()
    summary = tf.summary.merge_all()
    f.write(" Done\n")
    f.flush()

    f.write("Create writer...")
    f.flush()
    writer = tf.summary.FileWriter(output_path + "/tensorboard")
    writer.add_graph(session.graph)
    f.write(" Done\nEmbedding...")
    f.flush()

    embedding_amout = 1000
    # Make sprite and labels.
    sprite_path = output_path + '/../sprite.png'
    if not os.path.isfile(sprite_path):
        f.write('\n resize...')
        f.flush()
        images = np.zeros((embedding_amout, 96, 96, 3), np.float32)
        for i in range(0, embedding_amout, 100):
            images[i:i + 100] = ndimage.zoom(CNN.load(files_test[i:i+100]), (1, 96/424, 96/424, 1))
        #images = ndimage.zoom(CNN.load(files_test[:embedding_amout]), (1, 96/424, 96/424, 1))
        assert(images.shape == (embedding_amout, 96, 96, 3))
        f.write(' Done\n sprite...')
        f.flush()
        sprite = images_to_sprite(images)
        f.write(' Done\n')
        f.flush()
        scipy.misc.imsave(sprite_path, sprite)

    tsv_label_path = output_path + '/../labels.tsv'
    if not os.path.isfile(tsv_label_path):
        metadata_file = open(tsv_label_path, 'w')
        content = open(labels_path, 'r').read()
        content = '\n'.join(content.replace(',', '\t').split('\n')[:embedding_amout + 1])
        metadata_file.write(content)
        metadata_file.close()

    embedding_size = np.prod(cnn.embedding_input.get_shape().as_list()[1:])
    embedding = tf.Variable(tf.zeros([embedding_amout, embedding_size]), name="test_embedding")
    embedding_placeholder = tf.placeholder(tf.float32, embedding.get_shape())
    embedding_assignment = embedding.assign(embedding_placeholder)

    embedding_input_flatten = tf.reshape(cnn.embedding_input, [-1, embedding_size])

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = sprite_path
    embedding_config.metadata_path = tsv_label_path
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([96, 96])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    f.write(" Done\n")
    f.flush()

    saver = tf.train.Saver(max_to_keep=20)

    if not resume:
        fm.write("# iteration rmse_test rmse_train\n")
        fx.write("# iteration rmse_batch\n")

    if resume:
        f.write("Restore session...")
        f.flush()
        backup = sorted(glob.glob(output_path + '/iter/*.data.index'))[-1]
        backup = backup.rsplit('.', 1)[0] # remove .index
        resume_iter = int(backup.split('/')[-1].split('.')[0])
        saver.restore(session, backup)
        f.write(' Done\nBackup file : {}\n'.format(backup))
        f.flush()
    else:
        f.write("Initialize variables...")
        f.flush()
        resume_iter = 0
        session.run(tf.global_variables_initializer())
        f.write(" Done\n")
        f.flush()

    cnn.train_counter = resume_iter

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
        if (i // 1000) % 3 == 1:
            data = np.zeros((embedding_amout, embedding_size))
            for j in range(0, embedding_amout, 100):
                f.write('{}/{}\n'.format(j, embedding_amout))
                f.flush()
                data[j: j+100] = session.run(embedding_input_flatten, feed_dict={ cnn.tfx: CNN.load(files_test[j: j+100]) })
            session.run(embedding_assignment, feed_dict={ embedding_placeholder: data })

            save_path = saver.save(session, '{}/tensorboard/model.ckpt'.format(output_path), i)
            f.write('Model saved in file: {}\n'.format(save_path))

        save_path = saver.save(session, '{}/iter/{:06d}.data'.format(output_path, i))
        f.write('Model saved in file: {}\n'.format(save_path))

        rmse_test = predict_all(session, CNN, cnn, files_test, labels_test, f)
        rmse_train = predict_all(session, CNN, cnn, files_train[:len(files_test)], labels_train[:len(files_test)], f)

        fm.write("{} {:.8g} {:.8g}\n".format(i, rmse_test, rmse_train))
        fm.flush()

        f.write("     |  TEST    |  TRAIN\n")
        f.write("-----+----------+-------\n")
        f.write("rmse |  {: <8.4}|  {:.4}\n".format(rmse_test, rmse_train))
        f.flush()

        s = tf.Summary()
        s.value.add(tag="rmse_test", simple_value=rmse_test)
        s.value.add(tag="rmse_train", simple_value=rmse_train)
        writer.add_summary(s, i)

        #make_fits(files_test, labels_test == 1, ps_test > 0.5, output_path + '/iter', "_{:06d}".format(i))

    f.write("Start daemon...")
    f.flush()

    # Use a Queue to generate batches and train in parallel
    q = queue.Queue(20)  # batches in the queue

    def trainer():
        for i in range(resume_iter, resume_iter + n_iter + 1):
            t0 = time()

            rem = resume_iter + n_iter + 1 - i
            if q.qsize() < min(3, rem):
                while q.qsize() < min(20, rem):
                    sleep(0.05)

            xs, ys = q.get()
            t1 = time()

            if i % 100 == 0 and i != 0:
                f.write("Before the training\n")
                f.write("===================\n")
                f.flush()
                print_log(xs, ys)

            if i == 102 or i == 1002:
                from tensorflow.python.client import timeline
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                rmse, _ = cnn.train(session, xs, ys, options=run_options, run_metadata=run_metadata)
                # google chrome : chrome://tracing/

                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(output_path + '/timeline.json', 'w') as tlf:
                    tlf.write(ctf)
            elif i % 50 == 0:
                rmse, s = cnn.train(session, xs, ys, tensors=[summary])
                writer.add_summary(s[0], i)
            else:
                rmse, _ = cnn.train(session, xs, ys)

            fx.write('{}    {:.6} \n'.format(i, rmse))

            if i % 100 == 0 and i != 0:
                f.write("\nAfter the training\n")
                f.write("==================\n")
                print_log(xs, ys)
                fx.flush()

            if i % 1000 == 0 and i != 0:
                save_statistics(i)

            t2 = time()
            f.write('{:06d}: ({}) {: >6.3f}s+{:.3f}s {} rmse_batch={:.3f}\n'.format(
                i, q.qsize(), t1 - t0, t2 - t1, xs.shape, rmse))
            f.flush()

            q.task_done()

    t = threading.Thread(target=trainer)
    t.daemon = True
    t.start()

    f.write(" Done\nStart feeders...")
    f.flush()

    # the n+1
    xs, ys = CNN.batch(files_train, labels_train)
    q.put((xs, ys))

    n_feeders = 2
    assert n_iter % n_feeders == 0
    def feeder():
        for _ in range(n_iter // n_feeders):
            xs, ys = CNN.batch(files_train, labels_train)
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
