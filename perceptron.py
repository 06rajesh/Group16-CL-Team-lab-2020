import numpy as np
import time
from dictVectorizer import CustomDictVectorizer


class Perceptron:
    def __init__(self, name=None, features_dir="weights", activation=0.0, learning_rate=0.1, epochs=10):
        self._name = name
        self._activation = activation
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._weights = list()
        self._features_dir = features_dir

    def activation_function(self, score):
        """
        activate the prediction depending on the socre
        :param score: float, values computed by current weight
        :return: boolean
        """
        if score >= self._activation:
            return 1
        else:
            return 0

    def train(self, x, y):
        """
        Train Perceptron model
        :param x: list of feature dict created by POStoken class
        :param y: list of single pos class 1 if true or 0 if false
        :return: weights, dict of features list
        """

        w = np.zeros(301)
        for e in range(0, self._epochs):
            start = time.time()

            for i in range(len(x)):
                f_list = x[i]

                f = 0
                for j in range(len(f_list)):
                    f += w[j] * f_list[j]
                y_pred = self.activation_function(f)

                for j in range(len(f_list)):
                    w[j] = w[j] + self._learning_rate * (y[i] - y_pred) * f_list[j]

            completed = time.time() - start
            if self._name is None:
                print("Epochs {} completed in {}".format(e, completed))
            else:
                print("{} : Epochs {} completed in {}".format(self._name, e, completed))

        self._weights = w
        return w

    # def train(self, x, y):
    #     w = np.zeros(len(x[0]))
    #     y_hat = np.ones(len(y))
    #     # error = np.ones(len(y))
    #     # errors = list()
    #
    #     for e in range(0, self._epochs):
    #         start = time.time()
    #
    #         for n in range(0, len(x)):
    #             # dot product of weights and x(input)
    #             # triggers activation function if > 0
    #
    #             v = x[n]
    #             f = np.dot(v, w)
    #             y_pred = self.activation_function(f)
    #             y_hat[n] = y_pred
    #
    #             # Updating Weights
    #             for w_i in range(0, len(w)):
    #                 w[w_i] = w[w_i] + self._learning_rate * (y[n] - y_pred) * v[w_i]
    #
    #         completed = time.time() - start
    #         if self._name is None:
    #             print("Epochs {} completed in {}".format(e, completed))
    #         else:
    #             print("{} : Epochs {} completed in {}".format(self._name, e, completed))
    #
    #     self._weights = w
    #     return w

    def predict(self, x):
        weights = self._weights
        prediction = 0.

        if len(weights) != len(x):
            print("Weights mismatched, can not be predicted")
        else:
            prediction = np.dot(x, weights)

        return prediction
