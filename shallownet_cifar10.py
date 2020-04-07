from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimage.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from pyimage.plot.historyplot import HistoryPlot


print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("[INFO] training network...")
N_epoch = 40
H = model.fit(trainX, trainY, validation_data=(testX, testY),
                    batch_size=32, epochs=N_epoch, verbose=1)

print("[INFO] predict test data...")
pred = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            pred.argmax(axis=1),
                            target_names=labelNames))

hp = HistoryPlot(N_epoch)
hp.show(H)
