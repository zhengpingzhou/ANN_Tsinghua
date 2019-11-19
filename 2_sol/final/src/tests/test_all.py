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