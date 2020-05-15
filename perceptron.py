import numpy as np
from posToken import PosToken


class Perceptron:
    def __init__(self, activation=0.0, learning_rate=0.1, epochs=10):
        self._activation = activation
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._weights = list()
        # self._all_features_list = PosToken.all_features_list()

    # def convert_to_feature_vector(self, features_list=None):
    #     if features_list is None:
    #         features_list = []
    #     features = np.zeros(len(self._all_features_list))
    #     for feat in features_list:
    #         idx = self._all_features_list.index(feat)
    #         features[idx] = 1
    #     return features

    def activation_function(self, score):
        if score >= self._activation:
            return 1
        else:
            return 0

    def train(self, x, y):
        w = np.zeros(len(x[0]))
        y_hat = np.ones(len(y))
        error = np.ones(len(y))
        errors = list()

        for e in range(0, self._epochs):
            for n in range(0, len(x)):
                # dot product of weights and x(input)
                # triggers activation function if > 0
                v = x[n]
                f = np.dot(v, w)
                y_pred = self.activation_function(f)
                y_hat[n] = y_pred

                # Updating Weights
                for w_i in range(0, len(w)):
                    w[w_i] = w[w_i] + self._learning_rate * (y[n] - y_pred) * v[w_i]

        self._weights = w
        return w

    def predict(self, x):
        weights = self._weights
        prediction = 0.

        if len(weights) != len(x):
            print("Weights mismatched, can not be predicted")
        else:
            prediction = np.dot(x, weights)

        return prediction
