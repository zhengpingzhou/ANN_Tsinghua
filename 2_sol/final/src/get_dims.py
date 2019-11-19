import numpy as np

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