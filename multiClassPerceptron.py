from typing import Dict
import numpy as np
import operator
import pickle
import os.path

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
    def __init__(self, save_to="weights"):
        self._weights = dict()
        self._savePath = save_to

    def save_to_file(self):
        filename = os.path.join(self._savePath, "weights.pickle")
        print("Saving model to file {}".format(filename))
        print("===================================")
        outfile = open(filename, 'wb')
        pickle.dump(self._weights, outfile)
        outfile.close()

    def load_weights(self):
        filename = os.path.join(self._savePath, "weights.pickle")

        if len(self._weights) > 0:
            return self._weights
        elif os.path.isfile(filename):
            infile = open(filename, 'rb')
            weights = pickle.load(infile)
            infile.close()
            return weights
        else:
            return None

    def train(self, inputs: Dict[str, MultiClassItem]):
        """
        Train MultiClassPerceptron Model
        :param inputs: Dictionary consists of MulticlassItem Class
        :return: null
        """

        count = 0
        total = len(inputs)
        finished = list()
        weights = self.load_weights()

        if weights is not None:
            for pos in weights:
                self._weights[pos] = weights[pos]
                finished.append(pos)
                count += 1

        if len(finished) > 0:
            f_list = ", ".join(finished)
            print("Already Finished : {}".format(f_list))

        for key, val in inputs.items():
            if key not in finished:
                item = inputs.get(key)
                print("Training Perceptron for {}".format(key))
                prcptn = Perceptron()
                self._weights[key] = prcptn.train(item.X, item.Y)
                count += 1
                print("Training Perceptron for {} complete. {}/{}".format(key, count, total))
                print("===================================")
                self.save_to_file()

        print("Training Completed.")
        print("===================================")
        self.save_to_file()
        return

    def predict(self, inputs):
        """
        Predict POS tag based of previously trained weights
        :param inputs: array of inputs, numpy array
        :return: list of predictied tag
        """
        y_out = list()
        weights = self.load_weights()

        if weights is not None:
            for x in inputs:
                scores = dict()
                for key, val in weights.items():
                    w = weights.get(key)
                    scores[key] = np.dot(x, w)

                selected = max(scores.items(), key=operator.itemgetter(1))[0]
                y_out.append(selected)
            return y_out
        else:
            print("No Weights Found to predict")
            return None
