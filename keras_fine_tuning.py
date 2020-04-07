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
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

img_row, img_col = (224, 224)

def generate_model(mode):
    if mode == "full":
        # tune all layers
        model = VGG16(weights=None, input_tensor=Input(shape=(img_row, img_col, 3)), classes=3)
        # model.summary()
    elif mode == "classifier":
        # load the feature extraction layers
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(img_row, img_col, 3)))

        # add the classification layer, which is going to train
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(3, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)
        # model.summary()

        # freeze extraction layers
        for layer in baseModel.layers:
            layer.trainable = False
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    opt = SGD(lr=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


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

#model = generate_model("classifier")
model = generate_model("full")

print("[INFO] training network...")
N_epoch = 40
H = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=32, epochs=N_epoch, verbose=1)

print("[INFO] predict test data...")
pred = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            pred.argmax(axis=1),
                            target_names=['cat', 'dog', 'panda']))

hp = HistoryPlot(N_epoch)
hp.show(H)
