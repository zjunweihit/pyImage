from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimage.preprocessing.resize import ResizePreprocessor
from pyimage.preprocessing.imagetoarray import ImageToArrayPreprocessor
from pyimage.datasets.datasetloader import DatasetLoader
from pyimage.plot.historyplot import HistoryPlot
from keras.optimizers import SGD
from imutils import paths
import argparse

from keras.applications import VGG16
from keras.layers import Input

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

img_row, img_col = (224, 224)

print("[INFO] loading images...")
imageFiles = list(paths.list_images(args["dataset"]))
resize_p = ResizePreprocessor(img_row, img_col)
image2array_p = ImageToArrayPreprocessor()
loader = DatasetLoader(preprocessors=[resize_p, image2array_p])
(data, labels) = loader.load(imageFiles, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=17)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = VGG16(weights=None, input_tensor=Input(shape=(img_row, img_col, 3)), classes=3)
#model.summary()

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
N_epoch = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=32, epochs=N_epoch, verbose=1)

print("[INFO] predict test data...")
pred = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            pred.argmax(axis=1),
                            target_names=['cat', 'dog', 'panda']))

hp = HistoryPlot(N_epoch)
hp.show(H)
