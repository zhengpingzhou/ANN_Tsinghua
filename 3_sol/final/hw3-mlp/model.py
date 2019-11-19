# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28], name='x_')
        self.y_ = tf.placeholder(tf.int32, [None], name='y_')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        ##############################################################
        # implement input -- Linear -- BN -- ReLU -- Linear -- loss  #
        #     the 10-class prediction output is named as "logits"    #
        ##############################################################
        with tf.device('/cpu:0'):
            with tf.name_scope('model'):
                with tf.name_scope('linear1'):
                    self.num_hidden = 384
                    self.W1 = weight_variable([28*28, self.num_hidden], name='W1')
                    self.b1 = bias_variable([self.num_hidden], name='b1')
                    logits = tf.matmul(self.x_, self.W1) + self.b1

                with tf.name_scope('bn'):
                    logits = batch_normalization_layer(logits, is_train)

                self.summary_histogram = tf.summary.histogram('output_after_linear', logits)

                with tf.name_scope('relu'):
                    logits = tf.nn.relu(logits)

                with tf.name_scope('dropout'):
                    logits = tf.nn.dropout(logits, self.keep_prob)

                with tf.name_scope('linear2'):
                    self.W2 = weight_variable([self.num_hidden, 10], name='W2')
                    self.b2 = bias_variable([10], name='b2')
                    logits = tf.matmul(logits, self.W2) + self.b2
        ##############################################################

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        ##############################################################
        # Tensorboard visualization
        ##############################################################
        self.summary_loss = tf.summary.scalar('train_loss', self.loss)
        self.summary_acc = tf.summary.scalar('train_acc', self.acc)
        self.summary = tf.summary.merge([self.summary_loss, self.summary_acc, self.summary_histogram])
        ##############################################################

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape, name):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def batch_normalization_layer(inputs, isTrain=True):
    # Implemented the batch normalization func and applied it on fully-connected layers
    eps = 1e-20
    scale = tf.Variable(tf.ones(inputs.get_shape()[-1]), name='gamma')
    offset = tf.Variable(tf.zeros(inputs.get_shape()[-1]), name='beta')
    epoch_mean = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False, name='epoch_mean')
    epoch_var = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False, name='epoch_var')

    if isTrain:
        decay = 1 - 1e-1
        mean, var = tf.nn.moments(inputs, [0])        
        update_epoch_mean = tf.assign(epoch_mean, epoch_mean * decay + mean * (1 - decay))
        update_epoch_var = tf.assign(epoch_var, epoch_var * decay + var * (1 - decay))
        with tf.control_dependencies([update_epoch_mean, update_epoch_var]):
            return tf.nn.batch_normalization(inputs, mean, var, offset, scale, eps)
    else:
        return tf.nn.batch_normalization(inputs, epoch_mean, epoch_var, offset, scale, eps)
    
