import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_input = None
        self._saved_output = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _save_input_for_backward(self, input):
        self._saved_input = input

    def _save_output_for_backward(self, output):
        self._saved_output = output


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        # f(x) = max(0, x)
        self._save_input_for_backward(input)
        return np.maximum(0, input)

    def backward(self, grad_output):
        '''Your codes here'''
        # f'(x) = 0 if x <= 0 else 1
        return grad_output * np.array(self._saved_input > 0)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        # f(x) = 1 / (1 + exp(-x))
        output = 1 / (1 + np.exp(-input))
        self._save_output_for_backward(output)
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        # f'(x) = f(x)(1 - f(x))
        return grad_output * self._saved_output * (1 - self._saved_output)



class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        # u = x * W + b
        self._save_input_for_backward(input)
        return np.matmul(input, self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''
        grad_input  = np.matmul(grad_output, np.transpose(self.W))
        self.grad_W = np.matmul(np.transpose(self._saved_input), grad_output)
        self.grad_b = grad_output
        return grad_input

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
