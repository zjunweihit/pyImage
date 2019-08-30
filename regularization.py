from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from pyimage.preprocessing.resize import ResizePreprocessor
from pyimage.datasets.datasetloader import DatasetLoader
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")

images = list(paths.list_images(args["dataset"]))
p = ResizePreprocessor(32, 32)
loader = DatasetLoader(preprocessors=[p])
(data, labels) = loader.load(images, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=17)

for r in (None, "l1", "l2"):
    print("[INFO] training with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=100,
                          learning_rate="constant", eta0=0.01, random_state=17)
    model.fit(trainX, trainY)

    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}".format(r, acc * 100))
