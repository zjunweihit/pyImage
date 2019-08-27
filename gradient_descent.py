from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    # apply a step function to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# generate 2-class classification problem with 1000 data points.
# todo: make_blobs()
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1 to the last entry of X, allow to treat the bias as a parameter within W
# apply the "bias trick"
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=17)

print("[INFO] training...")
# already include bias in X, so does W
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # trainX(3x500) * error(500x1) gets gradient(3x1) for each W with b inside
    gradient = trainX.T.dot(error)
    W += -args["alpha"] * gradient

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch+1), loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
# it doesn't work with testY(500,1), which should be identical as testX[:0] (500,)
# draw class 1 and 2 by column 0, and 1
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
