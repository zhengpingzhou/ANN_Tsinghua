import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            #todo: implement the new_state calculation given inputs and state
            W = tf.get_variable('W', [FLAGS.embed_units + self._num_units, self._num_units])
            b = tf.get_variable('b', [self._num_units], initializer=tf.constant_initializer(0.0))    
            new_state = self._activation(tf.matmul(tf.concat([inputs, state], axis=1), W) + b)

        return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            W_gate = tf.get_variable('W_gate', [FLAGS.embed_units + self._num_units, self._num_units * 2])
            b_gate = tf.get_variable('b_gate', [self._num_units * 2], initializer=tf.constant_initializer(1.0))

            gates = tf.sigmoid(tf.matmul(tf.concat([inputs, state], axis=1), W_gate) + b_gate) 
            z, r = tf.split(gates, num_or_size_splits=2, axis=1)
            
            W_cand = tf.get_variable('W_cand', [FLAGS.embed_units + self._num_units, self._num_units])
            b_cand = tf.get_variable('b_cand', [self._num_units], initializer=tf.constant_initializer(0.0))
            cand_h = self._activation(tf.matmul(tf.concat([inputs, r * state], axis=1), W_cand) + b_cand)

            new_h = z * state + (1 - z) * cand_h
            
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.device('/cpu:0'):
            with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
                c, h = state
                #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
                #todo: implement the new_c, new_h calculation given inputs and state (c, h)
                W_f = tf.get_variable('W_f', [FLAGS.embed_units + self._num_units, self._num_units])
                b_f = tf.get_variable('b_f', [self._num_units], initializer=tf.constant_initializer(self._forget_bias))
                
                W_i = tf.get_variable('W_i', [FLAGS.embed_units + self._num_units, self._num_units])
                b_i = tf.get_variable('b_i', [self._num_units], initializer=tf.constant_initializer(0.0))

                W_o = tf.get_variable('W_o', [FLAGS.embed_units + self._num_units, self._num_units])
                b_o = tf.get_variable('b_o', [self._num_units], initializer=tf.constant_initializer(0.0))
                
                W_c = tf.get_variable('W_c', [FLAGS.embed_units + self._num_units, self._num_units])
                b_c = tf.get_variable('b_c', [self._num_units], initializer=tf.constant_initializer(0.0))

                f = tf.sigmoid(tf.matmul(tf.concat([inputs, h], axis=1), W_f) + b_f)
                i = tf.sigmoid(tf.matmul(tf.concat([inputs, h], axis=1), W_i) + b_i)
                o = tf.sigmoid(tf.matmul(tf.concat([inputs, h], axis=1), W_o) + b_o)

                cand_c = self._activation(tf.matmul(tf.concat([inputs, h], axis=1), W_c) + b_c)
                new_c = f * c + i * cand_c
                new_h = o * self._activation(new_c)
            
        return new_h, (new_c, new_h)
