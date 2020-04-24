
class Evaluation:
    def __init__(self, tagset, original, predicted):
        self.tagset = tagset
        self.original = original
        self.predicted = predicted
        self.classes = {}
        self.macro = {}
        self.micro = {}

    def calculate(self):
        for tag in self.tagset:
            if len(tag) > 0:
                self.classes[tag] = self.calculate_class_score(tag)

        self.macro = self.calculate_macro()
        self.micro = self.calculate_micro()

    # A macro-average will compute the metric independently for each class and then take the average (hence treating all
    # classes equally)
    def calculate_macro(self):
        total_pr = 0
        total_rc = 0
        count = 0

        for c in self.classes:
            score = self.classes[c]
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

    # micro-average will aggregate the contributions of all classes to compute the average metric.
    def calculate_micro(self):
        tp = 0
        fn = 0
        fp = 0

        for c in self.classes:
            score = self.classes[c]
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

        for i in self.original:
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

