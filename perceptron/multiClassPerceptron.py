from typing import Dict
import multiprocessing
import numpy as np
import operator
import pickle
import os.path
from perceptron import Perceptron
import embeddings


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


def run_perceptron_process(item: MultiClassItem, name, features_dir, return_dict):
    """
    Run Perceptron classifier for each item in a different process
    :param item: A MulticlassItem to run perceptron
    :param name: string, name of the parts of speech class
    :param features_dir: string, path where list of features is saved
    :param return_dict: multiprocessing shared dict, return the trained weight to multiPerceptronClass
                        using this dictionary
    """
    print("Training Perceptron for {}".format(name))
    prcptn = Perceptron(name=name, features_dir=features_dir)
    return_dict[name] = prcptn.train(item.X, item.Y)
    print("Training Perceptron for {} complete".format(name))
    print("===================================")


def chunkify(items, chunk_len):
    """
    create batches of Parts of speech classes
    :param items: list of strings/classes/tags
    :param chunk_len: number, number of chunk in each batch
    :return list, list of chunks, each chunk contains chunk_len number of strings
    """
    return [items[i:i+chunk_len] for i in range(0, len(items), chunk_len)]


class MultiClassPerceptron:
    def __init__(self, save_to="weights", n_process=4):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._weights = dict()
        self._savePath = os.path.join(dir_path, save_to)
        self._n_process = n_process
        self.inputs = Dict[str, MultiClassItem]

    def save_to_file(self):
        """
        Save weight to _savePath directory in weights.pickle file
        """
        filename = os.path.join(self._savePath, "weights.pickle")
        print("Saving model to file {}".format(filename))
        print("===================================")
        outfile = open(filename, 'wb')
        pickle.dump(self._weights, outfile)
        outfile.close()

    def load_weights(self):
        """
        load weight from weights.pickle file in _savePath directory
        :return weights, dict
        """
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

    def fit(self, sentences, sentence_pos, classes):
        """
            Prepare Items for MultiClassPerceptron Using MultiClassItem
            :param sentences: List of Sentences, which each is a list of tokens
            :param sentence_pos: List of Sentenece Pos, which each is list of POS tag
            :param classes: list of classes in data
            :return: list of MultiClassItem
            """
        inputs = dict()
        for pos in classes:
            inputs[pos] = MultiClassItem(pos)
        g = embeddings.Glove()
        g.load()

        X = list()
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                features = g.get_embedding_val(sentences[i][j], average_on_none=True)
                X.append(features)
                for k, v in inputs.items():
                    item = inputs.get(k)
                    if sentence_pos[i][j] == k:
                        item.Y.append(1.)
                    else:
                        item.Y.append(0.)

        for k, v in inputs.items():
            item = inputs.get(k)
            item.X = X

        self.inputs = inputs

    def train(self):
        """
        Train MultiClassPerceptron Model
        :param inputs: Dictionary consists of MulticlassItem Class
        :return: null
        """

        count = 0
        inputs = self.inputs
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

        classes = list(inputs.keys())
        remained = [x for x in classes if x not in finished]
        remained = chunkify(remained, self._n_process)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = list()

        # running multiple process to utilize all cores
        for batch in remained:
            for key in batch:
                item = inputs.get(key)
                p = multiprocessing.Process(name="Process_" + str(key), target=run_perceptron_process, args=(item, key, self._savePath, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            for key in batch:
                self._weights[key] = return_dict[key]
                count += 1
                print("Training Completed {}/{}".format(count, total))

            self.save_to_file()

        print("Training Completed.")
        print("===================================")
        self.save_to_file()
        return

    def predict(self, sentences):
        """
        Predict POS tag based of previously trained weights
        :param sentences: list of inputs, Featured dict created by PosToken Class
        :return: list of predictied tag
        """
        y_out = list()
        weights = self.load_weights()

        if weights is not None:
            g = embeddings.Glove()
            g.load()

            for s in sentences:
                sentence_pred = list()
                for word in s:
                    scores = dict()
                    x = g.get_embedding_val(word, average_on_none=True)
                    for key, val in weights.items():
                        w = weights.get(key)
                        scores[key] = np.dot(x, w)

                    selected = max(scores.items(), key=operator.itemgetter(1))[0]
                    sentence_pred.append(selected)
                y_out.append(sentence_pred)
            return y_out
        else:
            print("No Weights Found to predict")
            return None
