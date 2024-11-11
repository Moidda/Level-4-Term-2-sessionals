import numpy as np
import pandas as pd
import pickle
import sys
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from time import *


class Convolution:
    def __init__(self, n_kernels, kernel_dim, stride, padding, learning_rate=0.1, kernels=None, bias=None):
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.padding = padding
        self.kernels = kernels
        self.bias = bias
        self.input = None
        self.learning_rate = learning_rate

    def zeropad(self, input, padding):
        """
        :param input: array of 2D matrices
        :return: array of padded 2D matrices
        """
        n_channels, n_rows, n_cols = input.shape[0], input.shape[1], input.shape[2]
        n_rows_out = 2*padding + n_rows
        n_cols_out = 2*padding + n_cols
        output = np.zeros((n_channels, n_rows_out, n_cols_out))
        output[:, padding:padding+n_rows, padding:padding+n_cols] = np.copy(input)
        return output

    def dilate(self, input, len):
        """
        :param input: array of 2D matrices
        :param len: length of dilation
        :return: array of dilated 2D matrices
        """
        n_channels, n_rows, n_cols = input.shape
        n_rows_out = n_rows + len*(n_rows-1)
        n_cols_out = n_cols + len*(n_cols-1) 
        output = np.zeros((n_channels, n_rows_out, n_cols_out))
        output[:,::len+1,::len+1] = input
        return output

    def rotate180(self, input):
        """
        :param input: an array of 2D matrices (representing one kernel)
        :return: an array of 180-degree rotated 2D matrices
        """
        output = np.copy(input)
        for i in range(output.shape[0]):
            output[i] = np.rot90(output[i])
            output[i] = np.rot90(output[i])
        return output

    def forward(self, input):
        """
        :param input: shape(n_channels, n_rows, n_cols)
        :returns: shape(n_kernels, n_rows_out, n_cols_out)
        self.kernels.shape(n_kernels, n_channels, kernel_dim, kernel_dim)
        """
        self.input = self.zeropad(input, self.padding)
        n_channels, n_rows, n_cols = self.input.shape
        n_rows_out = (n_rows-self.kernel_dim) // self.stride + 1
        n_cols_out = (n_cols-self.kernel_dim) // self.stride + 1
        output = np.zeros((self.n_kernels, n_rows_out, n_cols_out))

        if self.kernels is None:
            self.kernels = np.random.randn(self.n_kernels, self.input.shape[0], self.kernel_dim, self.kernel_dim)
            # Xavier initialization
            for i in range(self.n_kernels):
                self.kernels[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(n_channels*self.kernel_dim*self.kernel_dim)), size=(n_channels, self.kernel_dim, self.kernel_dim))

        if self.bias is None:
            self.bias = np.zeros((self.n_kernels, 1))
        
        for k in range(self.n_kernels):
            for row, row_out in zip(range(0, n_rows, self.stride), range(n_rows_out)):
                for col, col_out in zip(range(0, n_cols, self.stride), range(n_cols_out)):
                    output[k,row_out,col_out] = np.sum(self.input[:,row:row+self.kernel_dim, col:col+self.kernel_dim] * self.kernels[k,:,:,:]) + self.bias[k]
        
        return output

    def backward(self, dz):
        """
        :param dz: output gradient, from next layer
        dx = input gradient, to be passed to previous layer
        dk = weights/kernel gradient, for learning the weights
        db = bias gradient, for learning the biases
        """
        n_channels, n_rows, n_cols = self.input.shape
        n_rows_out = (n_rows-self.kernel_dim) // self.stride + 1
        n_cols_out = (n_cols-self.kernel_dim) // self.stride + 1

        # Gradient with respect to input
        dx = np.zeros((n_channels, n_rows, n_cols))
        dz_sparsed = self.dilate(dz, self.stride-1)     
        dz_sparsed = self.zeropad(dz_sparsed, self.kernel_dim-1)   
        # dx = sum( conv(dz_sparsed,rotated_K) )
        for k in range(self.n_kernels):
            for row in range(n_rows):
                for col in range(n_cols):
                    kernel_rotated = self.rotate180(self.kernels[k])
                    cell_wise_mult = dz_sparsed[k, row:row+self.kernel_dim, col:col+self.kernel_dim] * kernel_rotated[:,:,:]
                    dx[:,row,col] += np.sum(cell_wise_mult, axis=(1,2))

        # calculate gradient_kernels
        dk = np.zeros(self.kernels.shape)
        dz_dilated = self.dilate(dz, self.stride-1)
        # dk = conv(input, dilated output gradient)
        for k in range(self.n_kernels):
            for row in range(self.kernel_dim):
                for col in range(self.kernel_dim):
                    bostu = self.input[:, row:row+dz_dilated.shape[1], col:col+dz_dilated.shape[2]] * dz_dilated[k]
                    bostu = np.sum(bostu, axis=(1,2))
                    dk[k,:,row,col] = bostu

        # calculate gradient_bias
        db = np.zeros((self.n_kernels, 1))
        for k in range(self.n_kernels):
            db[k,:] = np.sum(dz[k,:,:])

        # update parameters
        self.kernels -= self.learning_rate*dk
        for k in range(self.n_kernels):
            self.bias[k] -= self.learning_rate*db[k]
        return dx


class MaxPooling:
    def __init__(self, pool_dim, stride):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward(self, input):
        """
        :param input: array of 2D matrices
        :return: array of 2D matrices, each matrix maxpool filter applied
        """
        self.input = input
        n_channels, n_rows, n_cols = self.input.shape
        n_rows_out = (n_rows-self.pool_dim) // self.stride + 1
        n_cols_out = (n_cols-self.pool_dim) // self.stride + 1
        output = np.zeros((n_channels, n_rows_out, n_cols_out))
        for c in range(n_channels):
            for row, row_out in zip(range(0, n_rows, self.stride), range(n_rows_out)):
                for col, col_out in zip(range(0, n_cols, self.stride), range(n_cols_out)):
                    output[c,row_out,col_out] = np.max(self.input[c, row:row+self.pool_dim, col:col+self.pool_dim])

        return output

    def backward(self, dz):
        """
        :param dz: 
        """
        n_channels, n_rows, n_cols = dz.shape
        dx = np.zeros_like(self.input)
        for i in range(n_channels):
            for j in range(n_rows):
                for k in range(n_cols):
                    patch = self.input[i, j*self.stride:j*self.stride+self.pool_dim, k*self.stride:k*self.stride+self.pool_dim]
                    max_index = np.unravel_index(np.argmax(patch), patch.shape)
                    dx[i, j*self.stride:j*self.stride+self.pool_dim, k*self.stride:k*self.stride+self.pool_dim][max_index] = dz[i, j, k]
        return dx
        # n_channels, n_rows, n_cols = self.input.shape
        # n_rows_out = (n_rows-self.pool_dim) // self.stride + 1
        # n_cols_out = (n_cols-self.pool_dim) // self.stride + 1
        # dx = np.zeros(self.input.shape)
        # for c in range(n_channels):
        #     for row, row_out in zip(range(0, n_rows, self.stride), range(n_rows_out)):
        #         for col, col_out in zip(range(0, n_cols, self.stride), range(n_cols_out)):
        #             st = np.argmax(self.input[c, row:row+self.pool_dim, col:col+self.pool_dim])
        #             idx, idy = np.unravel_index(st, (self.pool_dim, self.pool_dim))
        #             dx[c, row+idx, col+idy] = dz[c, row_out, col_out]
        # return dx
        # assert not isinstance(dz, str)


class Relu:
    def __init__(self):
        pass

    def forward(self, input):
        """
        :param input: input to ReLU activation
        :returns: max(0,input)
        """
        self.input = input
        ret = np.copy(input)
        ret[ret<0] = 0
        return ret

    def backward(self, dz):
        dx = np.copy(dz)
        dx[self.input<0] = 0
        return dx


# When working with exponents, there's a danger of overflow errors if the base gets too large. 
# This can easily happen when working with the output of a linear layer. 
# To protect ourselves from this, we can use a trick described by Paul Panzer on StackOverflow. 
# Because softmax(x) = softmax(x - c) for any constant c, we can calculate 
# $\sigma(x)_i = \frac{exp(x_i-max(x))}{\sum_{j=1}^{N} exp(x_j - max(x))}$ instead

class Softmax:
    def __init__(self):
        pass

    def forward(self, input):   
        exp = np.exp(input-np.max(input), dtype=np.float64)
        self.output = exp/np.sum(exp)
        return self.output
    
    def backward(self, dz):
        return dz 


class Flatten:
    def __init__(self):
        pass

    def forward(self, input):
        """
        :param input: array of 2D matrices
        :returns: row vector of shape (1,)
        """
        self.input_shape = input.shape
        return np.reshape(input, (1, input.shape[0]*input.shape[1]*input.shape[2]))

    def backward(self, dz):
        """
        :returs: array of 2D matrices
        """
        return np.reshape(dz, self.input_shape)


class FullyConnected:
    def __init__(self, n_output, learning_rate=0.01):
        """
        :n_output: no of outputs
        :input: shape(1,n_input); dynamically assigned during forward
        :weights: shape(n_inputs, n_outputs)
        :bias: shape(1,n_outputs)
        """
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        self.input = None
        self.weights = None
        self.bias = None

    def setParams(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        """
        :input: shape(1, n_input)
        :weight: shape(n_input, n_output)
        :bias: shape(1, n_output)
        :return: shape(1, n_output)
        """
        self.input = input
        n_input = input.shape[1]

        if self.weights is None:
            # Xavier initialization
            self.weights = np.random.normal(loc=0, scale=np.sqrt(1./(n_input*self.n_output)), size=(n_input, self.n_output))

        if self.bias is None:
            self.bias = np.zeros((1, self.n_output))

        return np.dot(self.input, self.weights) + self.bias

    def backward(self, dz):
        """
        :dz: shape(1, n_output)
        :dw: shape(n_input, n_output)
        :dx: shape(1, n_input)
        """
        dw = np.dot(self.input.T, dz)           # (n_input,n_output) = (n_input,1) DOT (1,n_output)
        dx = np.dot(dz, self.weights.T)         # (1,n_input) = (1,n_output) DOT (n_output,n_input)
        db = np.sum(dz, axis=1, keepdims=True)  # (1,n_output) = (1,n_output)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return dx


def crossEntropyLoss(y_pred, y_true):
    """
    Computes the cross-entropy loss between the true labels and predicted probabilities.

    Parameters:
        y_true (np.array): A row matrix of shape (1, n_classes) representing the true labels.
        y_pred (np.array): A row matrix of shape (1, n_classes) representing the predicted probabilities.

    Returns:
        float: The cross-entropy loss.
    """
    true_class = np.argmax(y_true, axis=1)
    loss = -np.log(y_pred[0, true_class])
    return np.sum(loss)


def loadImage(image_name):
    """
    :param image_name: filename of the image
    :returns: array of 2D matrices (1 channels) shape(1,h,w)
    Returns the grayscaled pixel values
    """
    img = Image.open(image_name).convert('L')
    img = img.resize((32,32))
    data = np.asarray(img)
    data = np.reshape(data, (1,32,32))
    return data

def oneHotEncode(digit):
    output = np.zeros((1, 10))
    output[0,digit] = 1.0
    return output

def oneHotDecode(onehot):
    return np.argmax(onehot)

def readcsv(csv_file_name):
    df = pd.read_csv(csv_file_name)
    X = []
    Y = []
    for index, row in df.iterrows():
        image_file_name = './Dataset/' + row['database name'] + '/' + row['filename']
        digit = row['digit']
        data = loadImage(image_file_name)
        label = oneHotEncode(digit)
        X.append(data)
        Y.append(label)
    return np.asarray(X), np.asarray(Y)


class Network:
    def __init__(self):
        self.layers = []
    
    def addLayer(self, layer):
        self.layers.append(layer)

    def batchNormalize(self, X):
        return (X-np.mean(X)) / np.std(X)

    def train(self, X_train, Y_train, batch_size, epoch):
        """
        :param X_train: training data. 
            Each input is a grayscaled image of 32x32 pixels 
            shape(n_inputs, 1,32,32)
        :param Y_train: label of each training data
            Each label is one-hot encoded
            shape(n_inputs, 1,10)
        :patam batch_size: the number of inputs to be taken in a single batch
        :param epoch: epoch
        :returns:
        """
        # Number of input samples
        n_inputs = X_train.shape[0]

        for e in tqdm(range(epoch), desc='Epoch', position=0):
            loss_training = 0
            
            # Shuffle the training data
            perm = np.random.permutation(n_inputs)
            X_train = X_train[perm]
            Y_train = Y_train[perm]
            
            # Split training data into mini-batches
            for i in tqdm(range(0, n_inputs, batch_size), desc='Batch', position=1, leave=False):
                X_batch = X_train[i : np.minimum(i+batch_size, n_inputs)]
                Y_batch = Y_train[i : np.minimum(i+batch_size, n_inputs)]
                
                start_time_batch = time()
                # Take a single input X and respective label Y (one-hot-encoded)
                for X, Y in zip(X_batch, Y_batch):
                    # Forward through the layers
                    output = self.batchNormalize(X)
                    for layer in self.layers:
                        output = layer.forward(output)
            
                    loss_training += crossEntropyLoss(output, Y)

                    # Backward through the layers
                    output_gradient = (output - np.copy(Y)) / batch_size
                    for layer in reversed(self.layers):
                        output_gradient = layer.backward(output_gradient)
                
                end_time_batch = time()
                time_batch = end_time_batch-start_time_batch
                remaining_time = (n_inputs*epoch - n_inputs*e - i)/batch_size * time_batch
                hours, remainder = divmod(remaining_time, 3600)
                minutes, seconds = divmod(remainder, 60)

            loss_training /= n_inputs

            # Perform validation on the training dataset to get different metrics
            loss_validation, accuracy, f1, c_matrix = self.validate(X_train, Y_train)

            print("Epoch:{0:2d}/{1:2d} | Loss Training:{2:.5f} | Loss Validation:{3:.5f} | Accuracy:{4:.5f} | F1:{5:5f} | ETA:{6:2d}:{7:2d}:{8:2d}"
            .format(e, epoch, loss_training, loss_validation, accuracy, f1, int(hours), int(minutes), int(seconds)))
            print('---Confusion Matrix---\n' + str(c_matrix))


    def validate(self, X_test, Y_test):
        """
        Parameters:
            X_test:
            Y_test: array of row matrices
        """
        loss = 0
        Y_pred_list = []
        for X, Y in zip(X_test, Y_test):
            output = self.batchNormalize(X)
            # forward pass
            for layer in self.layers:
                output = layer.forward(output)
            loss += crossEntropyLoss(output, Y)            
            Y_pred_list.append(oneHotDecode(output))

        Y_test_list = list(map(oneHotDecode, Y_test[:,0,:]))
        assert len(Y_test_list) == len(Y_pred_list), str(len(Y_test_list)) + '!=' + str(len(Y_pred_list))
        
        loss /= X_test.shape[0]
        accuracy = accuracy_score(Y_test_list, Y_pred_list)
        f1 = f1_score(Y_test_list, Y_pred_list, average='macro')
        c_matrix = confusion_matrix(Y_test_list, Y_pred_list)

        return loss, accuracy, f1, c_matrix

    def predict(self, X):
        """
        Parameters:
            X: a single input

        Returns:
            d: a digit representing the prediction
        """
        output = self.batchNormalize(X)
        for layer in self.layers:
            output = layer.forward(output)
        return oneHotDecode(output)



# network = Network()
# network.addLayer(Convolution(n_kernels=6, kernel_dim=5, stride=1, padding=0))
# network.addLayer(Relu())
# network.addLayer(Flatten())
# network.addLayer(FullyConnected(n_output=84))
# network.addLayer(Relu())
# network.addLayer(FullyConnected(n_output=10))
# network.addLayer(Relu())
# network.addLayer(Softmax())

# X_train, Y_train = readcsv('./Dataset/training-b.csv')
# network.train(X_train, Y_train, 128, 5)

# pickle_file = open('1705085_model.pickle', 'wb')
# pickle.dump(network, pickle_file)
# pickle_file.close()