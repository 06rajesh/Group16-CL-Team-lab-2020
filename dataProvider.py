import csv


class DataProvider:

    def __init__(self, path):
        self.path = path

    def load_original_data(self):
        data = list()
        line_count = 0

        with open(self.path + '/dev.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if len(row) != 0:
                    data.append(row[1])
                else:
                    data.append('STT')
                line_count += 1

        print("Total {} original data loaded".format(line_count))
        print("==============================")
        classes = set(data)
        return data, classes

    def load_predicted_data(self):
        data = list()
        line_count = 0

        with open(self.path + '/dev-predicted.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                if len(row) != 0:
                    data.append(row[1])
                else:
                    data.append('STT')
                line_count += 1

        print("Total {} Predicted data loaded".format(line_count))
        print("==============================")
        classes = set(data)
        return data, classes

    def load_train_data(self):
        sentences = []
        sentences_pos = []

        sent_tmp = []
        sent_pos_tmp = []
        all_pos = []

        with open(self.path + '/train.col') as csv_file:
            count = 0
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for line in csv_reader:
                if len(line) > 0:
                    pos = line[1]
                    sent_tmp.append(line[0])
                    sent_pos_tmp.append(pos)
                    all_pos.append(pos)
                else:
                    sentences.append(sent_tmp)
                    sentences_pos.append(sent_pos_tmp)
                    sent_tmp = []
                    sent_pos_tmp = []

                count += 1

            classes = set(all_pos)

        print("Train Data Loaded")
        print("Total Classes: {}".format(len(classes)))
        print("Total Sentences: {}".format(len(sentences)))
        print("Total Tagged Data: {}".format(count))
        print("==============================")

        return sentences, sentences_pos, classes

    def load_test_data(self):
        x = list()
        y = list()

        with open(self.path + '/train.col') as csv_file:
            count = 0
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for line in csv_reader:
                if len(line) > 0:
                    x.append(line[0])
                    y.append(line[1])
                    count += 1

            print("Total {} Test Data loaded".format(count))
            print("==============================")

            classes = set(y)
            return x, y, classes

