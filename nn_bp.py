from pyimage.nn.neuralnetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork([2, 2, 1])
print(nn)

# OR dataset
X_or = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_or = np.array([ [0],    [1],    [1],    [1] ])

# AND dataset
X_and = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_and = np.array([ [0],    [0],    [0],    [1] ])

# XOR dataset
X_xor = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
y_xor = np.array([ [0],    [1],    [1],    [0] ])

def NN_Evaluation(X, y, datasetName):
    print("====================================================")
    print("[INFO] training backpropagation Network for '{}'".format(datasetName))
    nn = NeuralNetwork([2, 2, 1])
    nn.fit(X, y, epochs=20000, updatePerEpoch=2000)

    for (x, label) in zip(X, y):
        # predict is a ndarray.
        # in this case, only 1 class output, so get the predict[0][0]
        pred = nn.predict(x)[0][0]
        step_out = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground truth={}, pred={:.4f}, output={} - {}".format(
            x, label[0], pred, step_out, 'V' if (step_out == label[0]) else 'X'))

NN_Evaluation(X_or, y_or, "OR")
NN_Evaluation(X_and, y_and, "AND")
NN_Evaluation(X_xor, y_xor, "XOR")
