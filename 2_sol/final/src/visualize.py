import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    plt.show()
    

def visualize(output):
    h, w = output.shape[-2], output.shape[-1]
    output_vis = output.reshape([-1, h, w])
    vis_square(output_vis)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python visualize.py [image filename]'
        sys.exit(1)

    plt.rcParams['figure.figsize'] = (10, 10)        # large images
    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'              # use grayscale output rather than a (potentially misleading) color heatmap

    image_filename = sys.argv[1]
    image = plt.imread(image_filename)
    input = image.reshape([1, 1, 28, 28])

    latest_epoch = 0
    latest_timestamp = 'epoch' + str(latest_epoch)
    model = load_model('model/' + latest_timestamp + '.pkl')

    output = input
    for layer in model.layer_list:
        output = layer.forward(output)    

        if layer.name == 'relu1':
            visualize(output)

    prediction = np.argmax(output)
    print output
    print prediction