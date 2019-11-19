# 文件结构

- `hw3-mlp/`目录下为MLP相关代码；
- `hw3-cnn/`目录下为CNN相关代码。

# 使用说明

实验环境：`Windows 10 (bash) + Python 2.7 + TensorFlow 1.3.0`

- **训练**：将训练数据文件夹`MNIST_data/`拷贝至代码目录下，并在代码目录下运行

  ```sh
  python main.py --is_train
  ```

- **测试**：在代码目录下运行

  ```sh
  python main.py --is_train=False
  ```


# 修改说明

### MLP

###### hw3-mlp/main.py

加入`TensorBoard`可视化代码：

```python
def train_epoch(model, sess, X, y, epoch, summary_writer=None):
  # ...
  ##############################################################
  # Tensorboard visualization
  ##############################################################
  if summary_writer: 
  summary_writer.add_summary(summary, (epoch * (50000 / FLAGS.batch_size) + times))
  ##############################################################

with tf.Session() as sess:
  # ...
  ##################################################################################
  # Tensorboard visualization
  ##################################################################################
  summary_writer_train = tf.summary.FileWriter(FLAGS.logdir_train, graph=tf.get_default_graph())
  summary_writer_valid = tf.summary.FileWriter(FLAGS.logdir_valid, graph=tf.get_default_graph())
  ##################################################################################
```

###### hw3-mlp/model.py

实现MLP模型：

```python
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
```

实现`batch_normalization`：

```python
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
```

使用`TensorBoard`进行可视化输出：

```python
##############################################################
# Tensorboard visualization
##############################################################
self.summary_loss = tf.summary.scalar('train_loss', self.loss)
self.summary_acc = tf.summary.scalar('train_acc', self.acc)
self.summary = tf.summary.merge([self.summary_loss, self.summary_acc, self.summary_histogram])
##############################################################
```

### CNN

###### hw3-cnn/main.py

加入`TensorBoard`可视化代码：

```python
def train_epoch(model, sess, X, y, epoch, summary_writer=None):
  # ...
  ##############################################################
  # Tensorboard visualization
  ##############################################################
  if summary_writer: 
  summary_writer.add_summary(summary, (epoch * (50000 / FLAGS.batch_size) + times))
  ##############################################################

with tf.Session() as sess:
  # ...
  ##################################################################################
  # Tensorboard visualization
  ##################################################################################
  summary_writer_train = tf.summary.FileWriter(FLAGS.logdir_train, graph=tf.get_default_graph())
  summary_writer_valid = tf.summary.FileWriter(FLAGS.logdir_valid, graph=tf.get_default_graph())
  ##################################################################################
```

###### hw3-cnn/model.py

实现CNN模型：

```python
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
```

实现卷积层：

```python
def conv_layer(inputs, k, c_in, c_out):
  W = weight_variable([k, k, c_in, c_out], name='W')
  b = bias_variable([c_out], name='b')
  return tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME') + b
```

实现`batch_normalization`：

```python
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
```

使用`TensorBoard`进行可视化输出：

```python
##############################################################
# Tensorboard visualization
##############################################################
self.summary_loss = tf.summary.scalar('train_loss', self.loss)
self.summary_acc = tf.summary.scalar('train_acc', self.acc)
self.summary = tf.summary.merge([self.summary_loss, self.summary_acc])
##############################################################
```

