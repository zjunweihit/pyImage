from pyimage.nn.perceptron import Perceptron
import numpy as np

# OR dataset
X_or = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_or = np.array([ [0],    [1],    [1],    [1] ])

# AND dataset
X_and = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_and = np.array([ [0],    [0],    [0],    [1] ])

# XOR dataset
X_xor = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_xor = np.array([ [0],    [1],    [1],    [0] ])

def PerceptronEvaluation(X, y, datasetName):
    print("====================================================")
    print("[INFO] training perceptron for '{}'".format(datasetName))
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=50)

    print("[INFO] verify perceptron")
    for (x, label) in zip(X, y):
        pred = p.predict(x)
        print("[INFO] data={}, ground truth={}, predict={} -- {}".format(
            x, label, pred, 'V' if pred == label[0] else 'X'))

PerceptronEvaluation(X_or,  y_or,  'OR')
PerceptronEvaluation(X_and, y_and, 'AND')
PerceptronEvaluation(X_xor, y_xor, 'XOR')
