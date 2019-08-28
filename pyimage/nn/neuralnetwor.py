# Backpropagation network

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # a list of W for each layer, but the output layer which is number of class(like 1 for bitwise test
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # skip the last 2 layers
        for i in np.arange(0, len(layers) - 2):
            # the neural node i has layers[i] input,
            # has to produce layers[i+1] output for next layers
            w = np.random.rand(layers[i] + 1, layers[i+1] + 1)
            # normalize
            self.W.append(w / np.sqrt(layers[i]))

        # the last 2 layers: input connections need a bias, but the output doesn't
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # print network each layer number
        return "NeuralNetwork: {}".format(
            "-".join(str(num) for num in self.layers))


    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def sigmoid_deriv(self, x):
        # When you do BP network, always like to select an activation function
        # which is differentiable
        return x * (1 - x)


    def fit(self, X, y, epochs=1000, updatePerEpoch=100):
        # each x has a bias with correponding W vector.
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, label) in zip(X, y):
                self.fit_one(x, label)

            if epoch == 0 or (epoch + 1) % updatePerEpoch == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))


    def fit_one(self, x, y):
        # a list of the activation output of each layer
        # but the first one is the input feature vector x
        #
        # the first one is the original input x,
        # later items are the layer's output by product[i-1]*w[i]
        act_outputs = [np.atleast_2d(x)]

        # == feedfoward ==
        # go through each layer for the NN
        for layer in np.arange(0, len(self.W)):
            # x * w[1] = product[1], for the first layer, input layer
            # product[i-1] * w[i] = product[i], for other layers, the final output is same size as class number
            product = act_outputs[layer].dot(self.W[layer])

            # pass the product to the activation function, sigmoid() here
            act_output = self.sigmoid(product)

            # append the data as the input data for next layer computation
            act_outputs.append(act_output)

        # == backpropagation ==
        # the last one is the final output of the NN
        error = act_outputs[-1] - y
        # the list of deta for applying to backpropagation
        deltas = [error * self.sigmoid_deriv(act_outputs[-1])]

        for layer in np.arange(len(act_outputs) - 2, 0, -1):
            delta = deltas[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(act_outputs[layer])
            deltas.append(delta)

        # reverse the deltas
        deltas = deltas[::-1]

        # == update weight ==
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * act_outputs[layer].T.dot(deltas[layer])
