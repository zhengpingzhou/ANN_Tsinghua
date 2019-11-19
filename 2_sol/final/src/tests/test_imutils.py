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
