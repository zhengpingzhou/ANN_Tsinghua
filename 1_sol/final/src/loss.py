from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return 0.5 * np.mean(np.square(input - target), axis=1)

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        self.softmax = np.transpose(np.transpose(np.exp(input)) / np.exp(input).sum(axis=1))
        cross_entropy = -np.mean(np.sum(target * np.log(self.softmax), axis=1))
        return cross_entropy

    def backward(self, input, target):
        return (self.softmax - target) / len(input)