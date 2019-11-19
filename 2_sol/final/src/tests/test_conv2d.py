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
    
