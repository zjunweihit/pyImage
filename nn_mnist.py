from pyimage.nn.neuralnetwor import NeuralNetwork

import keras
from keras.datasets import mnist
from sklearn.metrics import classification_report

# the data, split between train and test sets
# we can use keras mnist dataset as ndarray for
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# test with part of them
x_train = x_train[0:500]
y_train = y_train[0:500]
x_test = x_test[0:100]
y_test = y_test[0:100]

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# scale values to a range of 0 to 1 by dividing 255
x_train /= 255.0
x_test /= 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to become one hot data for labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

nn = NeuralNetwork([x_train.shape[1], 32, 16, 10])
print(nn)
nn.fit(x_train, y_train, epochs=1000)
pred = nn.predict(x_test)
pred = pred.argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), pred))
