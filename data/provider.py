import csv
import os.path as path


class Provider:

    def __init__(self):
        self.path = path.dirname(path.abspath(__file__))

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

    @staticmethod
    def construct_sentences(lines):
        """
        Reconstruct Sentences from all the lines read by CSV reader
        :param lines, all lines read by CSV reader
        :return: list of Sentences, List of POS list for each sentences and total Classes/Tags
        """
        sentences = []
        sentences_pos = []

        sent_tmp = []
        sent_pos_tmp = []
        all_pos = []

        for line in lines:
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
        classes = set(all_pos)

        return sentences, sentences_pos, classes

    def load_train_data(self):
        """
        Load the training data from the train.col file in given directory
        :return: list of Sentences, List of POS list for each sentences and total Classes/Tags
        """

        with open(self.path + '/train.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            sentences, sentences_pos, classes = self.construct_sentences(csv_reader)

        # half = int(len(sentences)/2)
        print("Train Data Loaded")
        print("Total Classes: {}".format(len(classes)))
        print("Total Sentences: {}".format(len(sentences)))
        print("==============================")

        return sentences, sentences_pos, classes

    def load_test_data(self):
        """
        Load the testing data from the test.col file in given directory
        :return: list of Sentences, List of POS list for each sentences and total Classes/Tags
        """
        with open(self.path + '/test.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            sentences, sentences_pos, classes = self.construct_sentences(csv_reader)

        print("Test Data Loaded")
        print("Total Classes: {}".format(len(classes)))
        print("Total Sentences: {}".format(len(sentences)))
        print("==============================")
        return sentences, sentences_pos, classes
