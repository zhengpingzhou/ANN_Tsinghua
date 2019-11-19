# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        ######################################################################################################
        # implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss 
        #     the 10-class prediction output is named as "logits"             
        ######################################################################################################                               
        with tf.device('/cpu:0'):
            with tf.name_scope('model'):
                with tf.name_scope('conv1'):
                    logits = conv_layer(x, k=5, c_in=1, c_out=4)

                with tf.name_scope('bn1'):
                    logits = batch_normalization_layer(logits, is_train)

                with tf.name_scope('relu1'):
                    logits = tf.nn.relu(logits)

                with tf.name_scope('max_pool1'):
                    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                with tf.name_scope('conv2'):
                    logits = conv_layer(logits, k=7, c_in=4, c_out=8)

                with tf.name_scope('bn2'):
                    logits = batch_normalization_layer(logits, is_train)

                with tf.name_scope('relu2'):
                    logits = tf.nn.relu(logits)
                
                with tf.name_scope('max_pool2'):
                    logits = tf.nn.max_pool(logits, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                with tf.name_scope('linear'):
                    W_fc = weight_variable([7 * 7 * 8, 10], name='W')
                    b_fc = bias_variable([10], name='b')
                    logits = tf.reshape(logits, [-1, 7 * 7 * 8])
                    logits = tf.matmul(logits, W_fc) + b_fc

        ######################################################################################################

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        ##############################################################
        # Tensorboard visualization
        ##############################################################
        self.summary_loss = tf.summary.scalar('train_loss', self.loss)
        self.summary_acc = tf.summary.scalar('train_acc', self.acc)
        self.summary = tf.summary.merge([self.summary_loss, self.summary_acc])
        ##############################################################

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape, name):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv_layer(inputs, k, c_in, c_out):
    W = weight_variable([k, k, c_in, c_out], name='W')
    b = bias_variable([c_out], name='b')
    return tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME') + b

def batch_normalization_layer(inputs, isTrain=True):
    # Implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    eps = 1e-10
    scale = tf.Variable(tf.ones(inputs.get_shape()[-1]), name='gamma')
    offset = tf.Variable(tf.zeros(inputs.get_shape()[-1]), name='beta')
    epoch_mean = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False, name='epoch_mean')
    epoch_var = tf.Variable(tf.zeros(inputs.get_shape()[-1]), trainable=False, name='epoch_var')

    if isTrain:
        decay = 1 - 1e-3
        mean, var = tf.nn.moments(inputs, [0, 1, 2])        
        update_epoch_mean = tf.assign(epoch_mean, epoch_mean * decay + mean * (1 - decay))
        update_epoch_var = tf.assign(epoch_var, epoch_var * decay + var * (1 - decay))
        with tf.control_dependencies([update_epoch_mean, update_epoch_var]):
            return tf.nn.batch_normalization(inputs, mean, var, offset, scale, eps)
    else:
        return tf.nn.batch_normalization(inputs, epoch_mean, epoch_var, offset, scale, eps)