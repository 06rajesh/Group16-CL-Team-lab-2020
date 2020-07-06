from itertools import chain


class Evaluation:
    def __init__(self):
        self.classes = list()
        self.original = list()
        self.predicted = list()
        self.scores = {}

    @staticmethod
    def flatten(ini_list):
        flatten_list = list(chain.from_iterable(ini_list))
        return flatten_list

    def fit(self, classes, original, predicted):
        """
        fit the evaluation class to its original and predicted value,
        compute scores for each class
        """
        if isinstance(original[0], list):
            self.original = self.flatten(original)
        else:
            self.original = original

        if isinstance(predicted[0], list):
            self.predicted = self.flatten(predicted)
        else:
            self.predicted = predicted

        for tag in classes:
            if len(tag) > 0:
                self.scores[tag] = self.calculate_class_score(tag)

    def get_macro_score(self):
        """
        A macro-average will compute the metric independently for each class and then take the average (hence treating all
        classes equally)
        :return: dict of precision, recall and fscore
        """
        total_pr = 0
        total_rc = 0
        count = 0

        for c in self.scores:
            score = self.scores[c]
            total_pr += score['precision']
            total_rc += score['recall']
            count += 1

        precision = total_pr/count
        recall = total_rc/count

        fscore = self.get_f_score(precision, recall)

        return {
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        }

    def get_micro_score(self):
        """
        micro-average will aggregate the contributions of all classes to compute the average metric.
        :return: dict of precision, recall and fscore
        """
        tp = 0
        fn = 0
        fp = 0

        for c in self.scores:
            score = self.scores[c]
            tp += score['tp']
            fn += score['fn']
            fp += score['fp']

        precision = float(tp / float(tp + fp))
        recall = float(tp / float(tp + fn))

        fscore = self.get_f_score(precision, recall)

        return {
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        }

    def calculate_class_score(self, tag):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(self.predicted)):
            if self.original[i] == tag:
                if self.predicted[i] == tag:
                    tp += 1
                else:
                    fn += 1

            else:
                if self.predicted[i] == tag:
                    fp += 1
                else:
                    tn += 1

        if tp + fp == 0:
            precision = 0
        else:
            precision = float(tp / float(tp + fp))

        if tp + fn == 0:
            recall = 0
        else:
            recall = float(tp / float(tp + fn))

        fscore = self.get_f_score(precision, recall)

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'fscore': fscore
        }

    @staticmethod
    def get_f_score(precision, recall):
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * float(float(recall * precision) / float(recall + precision))

        return fscore
