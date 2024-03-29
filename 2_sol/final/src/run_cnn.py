from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
from visualize import save_model

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', in_channel=1, out_channel=4, kernel_size=5, pad=2, init_std=0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', kernel_size=2, pad=0))  # output shape: N x 8 x 14 x 14

model.add(Conv2D('conv2', in_channel=4, out_channel=8, kernel_size=5, pad=2, init_std=0.01))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', kernel_size=2, pad=0))  # output shape: N x 16 x 7 x 7

model.add(Reshape('flatten', (-1, 8 * 7 * 7)))
model.add(Linear('fc3', 8 * 7 * 7, 10, 0.1))

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


for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    timestamp = 'epoch' + str(epoch)
    save_model(model, 'model/' + timestamp + '.pkl')

    if epoch >= 5 and epoch % 5 == 0:
        config['learning_rate'] /= 2

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
