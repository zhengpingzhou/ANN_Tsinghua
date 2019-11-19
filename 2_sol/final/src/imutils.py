import numpy as np


def im2col(input, dims, mode='sliding'):
    n, c_in, _, h_in, h_out, w_in, w_out, k = dims

    h_idx, w_idx = _im2col_2d_idx(dims, mode)
    input_col = input[:, :, h_idx, w_idx]                       # [n, c_in, k * k, h_out * w_out]                                                                  
    input_col = np.transpose(input_col, [0, 3, 1, 2])           # [n, h_out * w_out, c_in, k * k]
    input_col = np.reshape(input_col, [n, h_out * w_out, -1])   # [n, h_out * w_out, c_in * k * k]
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
