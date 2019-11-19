import numpy as np
from get_dims import get_dims
from imutils import im2col, col2im


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