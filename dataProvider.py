import csv


class DataProvider:

    def __init__(self, path):
        self.words = {}
        self.original_label = {}
        self.predicted_label = {}
        self.tagset = set([])
        self.path = path

    def load_data(self):
        self.load_original_data()
        self.load_predicted_data()

    def load_original_data(self):
        with open(self.path + '/dev.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if len(row) != 0:
                    self.original_label[line_count] = row[1]
                    self.words[line_count] = row[0]
                    self.tagset.add(row[1])
                else:
                    self.original_label[line_count] = 'STT'
                    self.words[line_count] = '<S>'
                line_count += 1

    def load_predicted_data(self):
        with open(self.path + '/dev-predicted.col') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if len(row) != 0:
                    self.predicted_label[line_count] = row[1]
                else:
                    self.predicted_label[line_count] = 'STT'
                line_count += 1
