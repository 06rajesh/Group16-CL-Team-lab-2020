from typing import Dict
import numpy as np
import operator
import pickle

from perceptron import Perceptron
from posToken import PosToken


class MultiClassItem:
    """
    Item Provider class for MultiClassPerceptron
    X holds the training X element for Single Class Perceptron
    Y holds the training Y element for Single Class Perceptron
    """
    def __init__(self, label):
        self.label = label
        self.X = list()
        self.Y = list()


class MultiClassPerceptron:
    def __init__(self):
        self._weights = dict()

    def save_to_file(self, filename="weights.pickle"):
        print("Saving model to file {}".format(filename))
        print("===================================")
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
        """
        Train MultiClassPerceptron Model
        :param inputs: Dictionary consists of MulticlassItem Class
        :return: null
        """

        count = 0
        total = len(inputs)

        for key, val in inputs.items():
            item = inputs.get(key)
            print("Training Perceptron for {}".format(key))
            prcptn = Perceptron()
            print("Training Perceptron for {} complete. {}/{}".format(key, count, total))
            print("===================================")
            self._weights[key] = prcptn.train(item.X, item.Y)
            count += 1

        print("Training Completed.")
        print("===================================")
        self.save_to_file()
        return

    def predict(self, x: list):
        """
        Predict POS tag based of previously trained weights
        :param x: List of string, tokens
        :return: list of predictied tag
        """
        y_out = list()

        for term in x:
            weights = self.load_weights()
            token = PosToken(term)
            features = token.get_features()
            pcp = Perceptron()
            feature_vec = pcp.convert_to_feature_vector(features)
            scores = dict()
            for key, val in weights.items():
                w = weights.get(key)
                scores[key] = np.dot(feature_vec, w)

            selected = max(scores.items(), key=operator.itemgetter(1))[0]
            y_out.append(selected)
        return y_out
