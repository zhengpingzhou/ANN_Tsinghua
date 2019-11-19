from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'h1_dim': 392,
    'h2_dim': 196,
    'learning_rate': 0.1,
    'weight_decay': 0.0002,
    'momentum': 0.0001,
    'batch_size': 100,
    'max_epoch': 1000,
    'disp_freq': 100,
    'test_epoch': 1
}

# Your model defintion here
# You should explore different model architecture
model = Network()

''' Network with 2 hiden layers'''
model.add(Linear('fc1', 784, config['h1_dim'], 0.01))
model.add(Relu('h1'))
model.add(Linear('fc2', config['h1_dim'], config['h2_dim'], 0.01))
model.add(Relu('h2'))
model.add(Linear('fc3', config['h2_dim'], 10, 0.01))

''' Network with 1 hiden layer'''
# model.add(Linear('fc1', 784, config['h1_dim'], 0.01))
# model.add(Relu('h1'))
# model.add(Linear('fc2', config['h1_dim'], 10, 0.01))
loss = EuclideanLoss(name='loss')


for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
