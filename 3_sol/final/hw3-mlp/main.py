# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
from model import Model
from load_data import load_mnist_2d

tf.app.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 100, "number of epochs")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean("is_train", True, "False to inference")
tf.app.flags.DEFINE_string("data_dir", "./MNIST_data", "data dir")
tf.app.flags.DEFINE_string("train_dir", "./train", "training dir")
tf.app.flags.DEFINE_integer("inference_version", 0, "the version for inferencing")
tf.app.flags.DEFINE_string("logdir_train", "./summary/train", "logdir(train) for tensorboard")
tf.app.flags.DEFINE_string("logdir_valid", "./summary/valid", "logdir(validation) for tensorboard")
FLAGS = tf.app.flags.FLAGS


def shuffle(X, y, shuffle_parts):  # Shuffle the X and y
    chunk_size = len(X) / shuffle_parts
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train_epoch(model, sess, X, y, epoch, summary_writer=None): # Training Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: FLAGS.keep_prob}
        loss_, acc_, _, summary = sess.run([model.loss, model.acc, model.train_op, model.summary], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
        ##############################################################
        # Tensorboard visualization
        ##############################################################
        if summary_writer: 
            summary_writer.add_summary(summary, (epoch * (50000 / FLAGS.batch_size) + times))
        ##############################################################
    loss /= times
    acc /= times
    return acc, loss


def valid_epoch(model, sess, X, y, epoch, summary_writer=None):  # Valid Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed < len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch, model.keep_prob: 1.0}
        loss_, acc_, summary = sess.run([model.loss, model.acc, model.summary], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
        # ##############################################################
        # # Tensorboard visualization
        # ##############################################################
        if summary_writer: 
            summary_writer.add_summary(summary, ((epoch * (10000 / FLAGS.batch_size) + times)) * 5)
        # ##############################################################
    loss /= times
    acc /= times
    return acc, loss


def inference(model, sess, X):  # Test Process
    return sess.run([model.pred], {model.x_: X, model.keep_prob: 1.0})[0]


with tf.Session() as sess:
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if FLAGS.is_train:
        X_train, X_test, y_train, y_test = load_mnist_2d(FLAGS.data_dir)
        X_val, y_val = X_train[50000:], y_train[50000:]
        X_train, y_train = X_train[:50000], y_train[:50000]
        mlp_model = Model(True)
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            mlp_model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            tf.global_variables_initializer().run()

        ##################################################################################
        # Tensorboard visualization
        ##################################################################################
        summary_writer_train = tf.summary.FileWriter(FLAGS.logdir_train, graph=tf.get_default_graph())
        summary_writer_valid = tf.summary.FileWriter(FLAGS.logdir_valid, graph=tf.get_default_graph())
        ##################################################################################

        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            train_acc, train_loss = train_epoch(mlp_model, sess, X_train, y_train, epoch, summary_writer_train)  # Complete the training process
            X_train, y_train = shuffle(X_train, y_train, 1)

            val_acc, val_loss = valid_epoch(mlp_model, sess, X_val, y_val, epoch, summary_writer_valid)  # Complete the valid process

            if val_acc >= best_val_acc:  # when valid_accuracy > best_valid_accuracy, save the model
                best_val_acc = val_acc
                best_epoch = epoch + 1
                test_acc, test_loss = valid_epoch(mlp_model, sess, X_test, y_test, epoch)  # Complete the test process
                mlp_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=mlp_model.global_step)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(mlp_model.learning_rate.eval()))
            print("  training loss:                 " + str(train_loss))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            print("  test loss:                     " + str(test_loss))
            print("  test accuracy:                 " + str(test_acc))

            if train_loss > max(pre_losses):  # Learning rate decay
                sess.run(mlp_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [train_loss]

    else:
        mlp_model = Model(False)
        if FLAGS.inference_version == 0:  # Load the checkpoint
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        mlp_model.saver.restore(sess, model_path)
        X_train, X_test, y_train, y_test = load_mnist_2d(FLAGS.data_dir)  # load_mnist_2d when implementing MLP

        count = 0
        for i in range(len(X_test)):
            test_image = X_test[i].reshape((1, 784))  # May be different in MLP model
            result = inference(mlp_model, sess, test_image)[0]
            if result == y_test[i]:
                count += 1
        print("test accuracy: {}".format(float(count) / len(X_test)))
