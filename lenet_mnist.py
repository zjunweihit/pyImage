from pyimage.nn.conv.lenet import LeNet
from pyimage.plot.historyplot import HistoryPlot
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

import keras
from keras.datasets import mnist
from sklearn.metrics import classification_report

# the data, split between train and test sets
# we can use keras mnist dataset as ndarray for
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

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

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
pred = model.predict(x_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1),
                            pred.argmax(axis=1),
                            target_names=[str(x) for x in np.arange(0, 10)]))

hp = HistoryPlot("ggplot", 20)
hp.show(H)
