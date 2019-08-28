# correctly classify AND and OR, but fails at XOR
# w(t+1) = w(t) + alpha * (pred - y)*x
# input X*W -> step func -> pred -> W(t+1)=W(t) - alpha*error*x
#                           error=pred - label
# Note:
#   we must add bias in W matrix for training and prediction,
#   otherwise 'AND' will not be predicted correctly
#       [INFO] training perceptron for 'AND'
#       [INFO] verify perceptron
#       [INFO] data=[0 0], ground truth=[0], predict=0 -- V
#       [INFO] data=[0 1], ground truth=[0], predict=1 -- X
#       [INFO] data=[1 0], ground truth=[0], predict=1 -- X
#       [INFO] data=[1 1], ground truth=[1], predict=1 -- V

import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # N is same as the input X column, each line is an data vector
        # X = array([[0, 0],
        #            [0, 1],
        #            [1, 0],
        #            [1, 1]])
        # with bias
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        '''
        step function is like the activation function
        :param x:
        :return:
        '''
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        '''
        training the dataset
        :param X: training data
        :param y: label
        :param epochs:
        :return:
        '''
        # insert a column 1 at the end of the matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, label) in zip(X, y):
                # input features and weight matrix's dot product goes through the step function
                # to get the prediction
                pred = self.step(np.dot(x, self.W))
                if pred != label:
                    error = pred - label
                    self.W += -self.alpha * error * x

    def predict(self, X):
        # make sure input is a matrix
        X = np.atleast_2d(X)
        X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))
