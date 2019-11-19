# 文件结构

- `cell.py`：实现`BasicRNNCell`, `GRUCell`, `BasicLSTMCell`等基本单元；
- `model.py`：实现神经网络框架；
- `main.py`：实现数据读入、模型可视化等。

# 使用说明

**实验环境**：`Windows10 + Python 3.5 (64 bit) + TensorFlow 1.3.0`

**训练命令**：将包含了`train.txt`, `dev.txt`, `test.txt`, `vector.txt`的数据文件夹`data/`拷贝至代码目录下，并在代码目录下运行

```sh
$ python3 main.py
```

# 修改说明

###### cell.py

实现`BasicRNNCell`, `GRUCell`, `BasicLSTMCell`等基本单元：

 ```python
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

 ```

###### model.py

实现`placeholder`：

```python
#todo: implement placeholders
self.texts = tf.placeholder(shape=[None, None], dtype=tf.string)  # shape: batch*len
self.texts_length = tf.placeholder(shape=[None], dtype=tf.int64)  # shape: batch
self.labels = tf.placeholder(shape=[None], dtype=tf.int64)  # shape: batch
```

实现神经网络结构（将上面的部分解除注释，则为单层RNN的网络程序）：

```python
################################################################################################
# Single Layer RNN
################################################################################################
# if num_layers == 1:
#     cell = BasicRNNCell(num_units)
#     # cell = GRUCell(num_units)
#     # cell = BasicLSTMCell(num_units)
# outputs, state = dynamic_rnn(cell, self.embed_input, self.texts_length, dtype=tf.float32, scope="rnn")
# logits = tf.layers.dense(inputs=state, units=num_labels)
################################################################################################

#todo: implement unfinished networks
################################################################################################
# Final Bi-LSTM Network
################################################################################################
cell_fw = BasicLSTMCell(num_units)
cell_bw = BasicLSTMCell(num_units)
outputs, states, = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_input, dtype=tf.float32, scope='rnn')

(cell_fw, hidden_fw), (cell_bw, hidden_bw) = states
logits = tf.layers.dense(inputs=tf.concat([hidden_fw, hidden_bw], 1), units=num_labels)
################################################################################################
```

###### main.py

导入词向量：

```python
embed = []
embed_dict = {}
with open(os.path.join(FLAGS.data_dir, 'vector.txt'), 'r') as f:
    for line in f.readlines():
    tokens = line.split()
    embed_dict[tokens[0]] = [float(x) for x in tokens[1:]]

for vocab in vocab_list:
    if vocab in embed_dict:
    	embed.append(embed_dict[vocab])
    else:
    	embed.append([0.0] * FLAGS.embed_units)
```

加入TensorBoard可视化代码：

```python
loss, accuracy = evaluate(model, sess, data_dev)
print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
################################################################################################
summary_dev = tf.Summary()
summary_dev.value.add(tag='loss/dev', simple_value=loss)
summary_dev.value.add(tag='accuracy/dev', simple_value=accuracy) 
summary_writer.add_summary(summary_dev, epoch)
################################################################################################

loss, accuracy = evaluate(model, sess, data_test)
print("        test_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))
#todo: implement the tensorboard code recording the statistics of development and test set
################################################################################################
summary_test = tf.Summary()
summary_test.value.add(tag='loss/test', simple_value=loss)
summary_test.value.add(tag='accuracy/test', simple_value=accuracy) 
summary_writer.add_summary(summary_test, epoch)
################################################################################################
```

