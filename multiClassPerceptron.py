from typing import Dict
import numpy as np
import operator
import pickle

from perceptron import Perceptron
from posToken import PosToken


class MultiClassItem:
    def __init__(self, label):
        self.label = label
        self.X = list()
        self.Y = list()


class MultiClassPerceptron:
    def __init__(self):
        self._weights = dict()

    def save_to_file(self, filename="weights.pickle"):
        outfile = open(filename, 'wb')
        pickle.dump(self._weights, outfile)
        outfile.close()

    def load_weights(self, filename="weights.pickle"):
        if len(self._weights) > 0:
            return self._weights
        else:
            infile = open(filename, 'rb')
            weights = pickle.load(infile)
            infile.close()
            return weights

    def train(self, inputs: Dict[str, MultiClassItem]):
        for key, val in inputs.items():
            item = inputs.get(key)
            print("Training Perceptron for {}".format(key))
            prcptn = Perceptron()
            print("Training Perceptron for {} complete".format(key))
            print("===================================")
            self._weights[key] = prcptn.train(item.X, item.Y)

        self.save_to_file()
        return

    def predict(self, term: str):
        weights = self.load_weights()
        token = PosToken(term)
        features = token.get_features()
        scores = dict()
        for key, val in weights.items():
            w = weights.get(key)
            scores[key] = np.dot(features, w)

        selected = max(scores.items(), key=operator.itemgetter(1))[0]
        return selected
