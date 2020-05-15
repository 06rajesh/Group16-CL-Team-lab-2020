from threading import Thread
import time
from multiClassPerceptron import MultiClassItem
from dictVectorizer import CustomDictVectorizer


class PerceptronThread(Thread):
    def __init__(self, threadID, name, item: MultiClassItem):
        Thread.__init__(self)
        self.exitFlag = 0
        self.threadID = threadID
        self.name = name
        self.item = item
        self._return = None

    def run(self):
        print("Starting {}".format(self.name))
        dv = CustomDictVectorizer()
        feature_vec = dv.transform(self.item.X)
        print("{}: {}".format(self.name, feature_vec.shape))
        print("Exiting {}".format(self.name))
        self._return = feature_vec

    def join(self, *args):
        Thread.join(self)
        return self._return


class ItemsFeatures:
    def __init__(self, items: dict):
        self.items = items

    def fit(self):
        items = dict()

        for idx, key in enumerate(self.items):
            thread = PerceptronThread(idx, key, self.items[key])
            thread.start()
            items[key] = thread.join()

        print(len(items['NNS']))
