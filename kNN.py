from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse

from pyimage.preprocessing.resize import ResizePreprocessor
from pyimage.datasets.datasetloader import DatasetLoader

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=3,
                help="# of nearest neighbors for classification")
#ap.add_argument("-j", "--jobs", type=int, default=-1,
#                help="# of jobs for kNN distance(-1 uses all available cores)")
args = vars(ap.parse_args())

# debug options
#IMG_PATH = "/home/zjunwei/gitroot/my/dataset/animals"
#imagePaths = list(paths.list_images(IMG_PATH))

print("[INFO] loading images...")
# in the same folder the images will be listed out of order
imagePaths = list(paths.list_images(args["dataset"]))

p = ResizePreprocessor(32, 32)
loader = DatasetLoader(preprocessors=[p])
(data, labels) = loader.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072)) # 32*32*3

print("[INFO] data size: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition data: 75% for training, 25% for test
#   it's not exactly 25% in number
#   if random_state=17, 236 cats, 257 dogs, 257 panda
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=17)

metric_list = ['minkowski', # default
               'manhattan', # L1 distance
               'euclidean', # L2 distance
               ]
for m in metric_list:
    print("[INFO] training kNN classifier (metric:{})...".format(m))
    model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=-1, metric=m)
    model.fit(trainX, trainY)

    print("[INFO] produce the test report...")
    report = classification_report(testY, model.predict(testX),
                                   target_names=le.classes_)
    print(report)

#model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=-1, metric='euclidean')
#model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=-1, metric='manhattan')
