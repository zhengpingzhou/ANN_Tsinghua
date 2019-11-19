# 文件结构

- `images/` 文件夹下为可视化时测试用图片;
- `tests/` 文件夹下为代码的部分单元测试；
- 根目录下为本次实验的CNN训练代码。

# 使用说明

实验环境：`Windows10 + Python2.7 (numpy, matplotlib)  `

- **训练**：将含有训练数据的`data/`文件夹拷贝至根目录下，并在根目录下运行`python run_cnn.py`
- **代码单元测试**：在`tests/`目录下运行`python test_all.py`，为笔者调试时所用。

# 修改说明

### 模型实现

###### functions.py

补全函数`conv2d_forward`, `conv2d_backward`, `avgpool2d_forward`, `avgpool2d_backward` ：

```python
def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    input_pad = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)
    dims = n, c_in, c_out, h_in, h_out, w_in, w_out, k = get_dims((input_pad, W), 'conv')

    '''im2col for input & W'''
    input_col = im2col(input_pad, dims, 'sliding')                      # [n, h_out * w_out, c_in * k * k]
    W_col = np.reshape(W, [c_out, c_in * k * k])                        # [c_out, c_in * k * k]

    '''ouput <- dot(w_col, input_col.T) + b, then transpose & reshape.'''
    output = np.matmul(input_col, W_col.T)                              # [n, h_out * w_out, c_out]
    output = output.transpose([0, 2, 1]).reshape([n, c_out, h_out, w_out])
    output += np.expand_dims(np.expand_dims(b, axis=1), axis=2)    
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = n (#sample) x c_out (#output channel) x h_out x w_out
        grad_b: gradient of b, shape = c_out
    '''
    input_pad = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)
    dims = n, c_in, c_out, h_in, h_out, w_in, w_out, k = get_dims((input_pad, W), 'conv')

    '''Inverse operation of transpose & reshape in conv2d_forward.'''
    grad_output = grad_output.reshape([n, c_out, h_out * w_out]).transpose([0, 2, 1])

    '''grad_input <- dot(grad_output, W_col), then col2im & trim the pads.'''
    W_col = np.reshape(W, [c_out, c_in * k * k])
    grad_input_col = np.matmul(grad_output, W_col)
    grad_input = col2im(grad_input_col, dims, 'sliding')
    grad_input = grad_input[:, :, pad:-pad, pad:-pad] if pad > 0 else grad_input

    '''grad_W <- dot(input_col.T, grad_output), then sum across batches & col2im.'''
    input_col = im2col(input_pad, dims, 'sliding')
    grad_W_col = np.matmul(np.transpose(input_col, [0, 2, 1]), grad_output).sum(axis=0).T
    grad_W = grad_W_col.reshape([c_out, c_in, k, k])

    '''grad_b <- grad_output, then sum across batches and channels.'''
    grad_b = np.sum(grad_output, axis=(0, 1))
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the w_indow to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    input_pad = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)
    dims = n, c_in, _, h_in, h_out, w_in, w_out, k = get_dims((input_pad, kernel_size), 'pooling')

    input_patch = np.reshape(input_pad, [n, c_in, h_in / k, k, w_in / k, k])
    output = input_patch.mean(axis=3).mean(axis=4)
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the w_indow to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    k = kernel_size
    grad_input = grad_output.repeat(k, axis=2).repeat(k, axis=3) * (1.0 / k / k)
    grad_input = grad_input[:, :, pad:-pad, pad:-pad] if pad > 0 else grad_input
    return grad_input
```

###### imutils.py

实现函数`im2col`, `col2im`, `_im2col_idx_2d`，分别用于实现im2col变换、逆变换、坐标计算：

```python
def im2col(input, dims, mode='sliding'):
    n, c_in, _, h_in, h_out, w_in, w_out, k = dims

    h_idx, w_idx = _im2col_2d_idx(dims, mode)
    input_col = input[:, :, h_idx, w_idx]                     # [n, c_in, k * k, h_out * w_out]                                                                  
    input_col = np.transpose(input_col, [0, 3, 1, 2])         # [n, h_out * w_out, c_in, k * k]
    input_col = np.reshape(input_col, [n, h_out * w_out, -1]) # [n, h_out * w_out, c_in * k * k]
    return input_col


def col2im(input_col, dims, mode='sliding'):
    n, c_in, _, h_in, h_out, w_in, w_out, k = dims

    input_col = np.reshape(input_col, [n, h_out * w_out, c_in, k * k])
    input_col = np.transpose(input_col, [0, 2, 3, 1])
    
    h_idx, w_idx = _im2col_2d_idx(dims, mode)
    input = np.zeros([n, c_in, h_in, w_in])
    np.add.at(input, (slice(None), slice(None), h_idx, w_idx), input_col)
    return input


def _im2col_2d_idx(dims, mode='sliding'):
    _, _, _, h_in, h_out, w_in, w_out, k = dims

    start_idx = np.arange(k)[:, None] * w_in + np.arange(k)
    if mode == 'sliding':
        offset_idx = np.arange(h_out)[:, None] * w_in + np.arange(w_out)   
    elif mode == 'distinct':
        offset_idx = np.arange(h_out * k, step=k)[:, None] * w_in + np.arange(w_out * k, step=k)
    else:
        raise ValueError('Mode must be either "sliding" or "distinct".')
    im2col_2d_idx = start_idx.ravel()[:, None] + offset_idx.ravel()

    h_idx = im2col_2d_idx / w_in
    w_idx = im2col_2d_idx % w_in    
    return h_idx, w_idx  

```

###### get_dims.py

实现函数`get_dims`，用于简化代码中各维度参数传递：

```python
def get_dims(args, mode='conv'):
    if mode == 'conv':
        input, W = args
        n, c_in, h_in, w_in = input.shape
        c_out, c_in, k, k = W.shape
        h_out, w_out = h_in - k + 1, w_in - k + 1
    elif mode == 'pooling':
        input, k = args
        n, c_in, h_in, w_in = input.shape
        h_out, w_out = h_in / k, w_in / k
        c_out = -1
    else:
        raise ValueError('Mode must be either "conv" or "pooling".')

    return n, c_in, c_out, h_in, h_out, w_in, w_out, k
```

###### loss.py

实现`SoftmaxCrossEntropyLoss`的`forward`及`backward`，用于计算交叉熵损失函数：

```python
class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        input -= np.max(input, axis=1)[:, np.newaxis]
        self.softmax = np.transpose(np.transpose(np.exp(input)) / np.exp(input).sum(axis=1))
        cross_entropy = -np.mean(np.sum(target * np.log(self.softmax), axis=1))
        return cross_entropy

    def backward(self, input, target):
        return (self.softmax - target) / len(input)
```

###### run_cnn.py

调节模型结构及训练超参数：

```python
# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', in_channel=1, out_channel=4, kernel_size=5, pad=2, init_std=0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', kernel_size=2, pad=0))  # output shape: N x 8 x 14 x 14

model.add(Conv2D('conv2', in_channel=4, out_channel=4, kernel_size=5, pad=2, init_std=0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', kernel_size=2, pad=0))  # output shape: N x 16 x 7 x 7

model.add(Reshape('flatten', (-1, 4 * 7 * 7)))
model.add(Linear('fc3', 4 * 7 * 7, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 2e-4,
    'momentum': 1e-4,
    'batch_size': 50,
    'max_epoch': 100,
    'disp_freq': 5,
    'test_epoch': 1
}
```

###### visualize.py

实现卷积层输出的可视化：

```python
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.show()
    

def visualize(output):
    h, w = output.shape[-2], output.shape[-1]
    output_vis = output.reshape([-1, h, w])
    vis_square(output_vis)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python visualize.py [image filename]'
        sys.exit(1)

    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'              # use grayscale output rather than a (potentially misleading) color heatmap

    image_filename = sys.argv[1]
    image = plt.imread(image_filename)
    input = image.reshape([1, 1, 28, 28])

    latest_epoch = 0
    latest_timestamp = 'epoch_44_' + str(latest_epoch)
    model = load_model('model/' + latest_timestamp + '.pkl')

    output = input
    for layer in model.layer_list:
        output = layer.forward(output)    

        if layer.name == 'relu1':
            visualize(output)

    prediction = np.argmax(output)
    print output
    print prediction
```

###  单元测试

###### tests/test_all.py

对`conv2d_forward`及`im2col, col2im` 函数进行单元测试：

```python
from random import randint
from test_conv2d import test_conv2d
from test_imutils import test_imutils


def make_dims(n, c_in, c_out, h_in, w_in, k):
    h_out = h_in - k + 1
    w_out = w_in - k + 1
    dims = n, c_in, c_out, h_in, h_out, w_in, w_out, k
    return dims


if __name__ == '__main__':
    print 'Testing conv2d_forward...'
    for i in range(10):
        test_conv2d(make_dims(
            n=randint(1, 5),
            c_in=randint(1, 5),
            c_out=randint(3, 5),
            h_in=randint(10, 20),
            w_in=randint(10, 20),
            k=randint(2, 5)
        ))

    print 'Testing imutils...'
    for i in range(10):
        test_imutils(make_dims(
            n=randint(1, 5),
            c_in=randint(1, 5),
            c_out=randint(3, 5),
            h_in=randint(10, 20),
            w_in=randint(10, 20),
            k=randint(2, 5)
        ))
```

###### tests/test_conv2d.py

对`conv2d_forward` 函数进行单元测试：

```python
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from functions import conv2d_forward


def test_conv2d(dims):
    n, c_in, c_out, h_in, h_out, w_in, w_out, k = dims

    input = np.arange(n * c_in * h_in * w_in)
    input = np.reshape(input, [n, c_in, h_in, w_in])
    
    W = np.arange(c_out * c_in * k * k)
    W = np.reshape(W, [c_out, c_in, k, k])

    expected = np.zeros([n, c_out, h_out, w_out])
    for i in xrange(n):
        for h in xrange(h_out):
            for w in xrange(w_out):
                for x in xrange(k):
                    for y in xrange(k):
                        for m in xrange(c_out):
                            for d in xrange(c_in):
                                expected[i][m][h][w] += input[i][d][h + x][w + y] * W[m][d][x][y]

    output = conv2d_forward(input, W, b=np.zeros([c_out], dtype=int), kernel_size=k, pad=0)
    assert np.all(expected - output == 0)
    print 'Test passed!'
    

```

###### tests/test_imutils.py

对`im2col`及`col2im` 函数进行单元测试：

```python
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from imutils import im2col, col2im


def test_imutils(dims):
    n, c_in, c_out, h_in, h_out, w_in, w_out, k = dims

    input = np.arange(n * c_in * h_in * w_in)
    input = np.reshape(input, [n, c_in, h_in, w_in])
    
    W = np.arange(c_out * c_in * k * k)
    W = np.reshape(W, [c_out, c_in, k, k])

    input_col = im2col(input, dims)
    input_1 = col2im(input_col, dims)

    divisor_h = np.concatenate((
        np.arange(k) + 1,
        np.ones(h_in - 2 * k) * k,
        np.arange(k)[::-1] + 1
    ))
    divisor_w = np.concatenate((
        np.arange(k) + 1,
        np.ones(w_in - 2 * k) * k,
        np.arange(k)[::-1] + 1
    ))
    divisor = np.dot(divisor_h.reshape(-1, 1), divisor_w.reshape(1, -1))

    assert np.all(input == input_1 / divisor)
    print 'Test passed!'

```

### im2col版本AvgPooling

笔者曾尝试使用`im2col`函数的`'distinct'` 模式实现AvgPooling，但最终构思出了更简洁的实现方式。这部分实验代码在这里给出：

```python
def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the w_indow to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    input_pad = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)
    dims = n, c_in, _, h_in, h_out, w_in, w_out, k = get_dims((input_pad, kernel_size), 'pooling')

    input_col = im2col(input_pad, dims, 'distinct')                         # [n, h_out * w_out, c_in * k * k]
    input_col = np.reshape(input_col, [n, h_out * w_out, c_in, k * k])      # [n, h_out * w_out, c_in, k * k]
    output_col = np.mean(input_col, axis=3)                                 # [n, h_out * w_out, c_in]
    output_col = np.transpose(output_col, [0, 2, 1])                        # [n, c_in, h_out * w_out]
    output = np.reshape(output_col, [n, c_in, h_out, w_out])                # [n, c_in, h_out, w_out]
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the w_indow to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    input_pad = np.pad(input, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant', constant_values=0)
    dims = n, c_in, _, h_in, h_out, w_in, w_out, k = get_dims((input_pad, kernel_size), 'pooling')

    '''Inverse operation of transpose & reshape in avgpool2d_forward.'''
    grad_output = np.reshape(grad_output, [n, c_in, h_out * w_out])                 # [n, c_in, h_out * w_out]
    grad_output = np.transpose(grad_output, [0, 2, 1])                              # [n, h_out * w_out, c_in]

    grad_input_col = np.repeat(grad_output, k * k, axis=2) * (1.0 / k / k)          # [n, h_out * w_out, c_in * k * k]
    grad_input = col2im(grad_input_col, dims, 'distinct')
    return grad_input

```

