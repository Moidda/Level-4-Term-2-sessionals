import sys
import os
from train_1705085 import *


pickle_file = open('1705085_model.pickle', 'rb')
network = pickle.load(pickle_file)
pickle_file.close()


# Lenet architecture
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



pred_file_content = 'FileName,Digit\n'
folder_name = '../test-b1/'
for entry in os.scandir(folder_name):
    if entry.is_file():
        image_path = folder_name + entry.name
        X = loadImage(image_path)
        prediction = network.predict(X)
        pred_file_content += entry.name + ',' + str(prediction) + '\n'

pred_file_name = '170585_prediction.csv'
pred_file = open(pred_file_name, 'w')
pred_file.write(pred_file_content)
pred_file.close()