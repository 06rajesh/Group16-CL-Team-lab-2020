import numpy as np
import os.path
import pickle


class CustomDictVectorizer:
    """
    Class to turn featured dict key and value into a unique feature
    and maintain, save feature list
    """
    def __init__(self, save_to="weights"):
        self.feature_list = list()
        self._savePath = save_to

    def save_to_file(self):
        """
        Save features to _savePath directory in features.pickle file
        """
        filename = os.path.join(self._savePath, "features.pickle")
        print("Saving model to file {}".format(filename))
        print("===================================")
        outfile = open(filename, 'wb')
        pickle.dump(self.feature_list, outfile)
        outfile.close()

    def load_features(self):
        """
        load weight from features.pickle file in _savePath directory
        :return features, list
        """
        filename = os.path.join(self._savePath, "features.pickle")
        if len(self.feature_list) > 0:
            return self.feature_list
        elif os.path.isfile(filename):
            infile = open(filename, 'rb')
            features = pickle.load(infile)
            infile.close()
            return features
        else:
            return None

    def fit(self, inputs: list, min_occurs=200):
        """
        create features list from all the provided features, converts feature name and value into a unique feature
        :param inputs: list of features where each item is a dictionary
        :param min_occurs: number, number of minimum count of a feature in total data
        :return: list, features
        """
        all_features = list()
        feature_count = dict()
        for features in inputs:
            for key in features:
                if type(features[key]) is str:
                    feature_name = key + '=' + features[key]
                else:
                    feature_name = key

                if feature_name in feature_count:
                    feature_count[feature_name] += 1
                else:
                    feature_count[feature_name] = 1

        feature_count = {k: v for k, v in sorted(feature_count.items(), key=lambda item: item[1], reverse=True)}

        for c in feature_count:
            if feature_count[c] > min_occurs:
                all_features.append(c)
            else:
                break

        self.feature_list = all_features
        print("Feature Vector length {}".format(len(all_features)))
        self.save_to_file()
        return all_features

    def get_features_length(self):
        """
        :return: length of the features list / feature vector
        """
        features = self.load_features()
        if features is not None:
            return len(features) + 1    # additional 1 for the bias
        else:
            return None

    def get_idx_values(self, f_list):
        """
        calculate feature indexes from the features dict created by PosToken class
        :param f_list: dict, dictionary of features with feature key and value
        :return: features index list in features list and feature values
        """
        indexes = list()
        values = list()
        feature_list = self.load_features()

        indexes.append(0)
        values.append(1.)

        if feature_list is not None:
            for f in f_list:
                if type(f_list[f]) is str:
                    feature_name = f + '=' + f_list[f]
                else:
                    feature_name = f
                try:
                    idx = feature_list.index(feature_name) + 1  # shifting for bias value
                except ValueError:
                    idx = -1
                indexes.append(idx)
                if idx != -1:
                    if type(f_list[f]) is int or type(f_list[f]) is float:
                        values.append(f_list[f])
                    else:
                        values.append(1.)
                else:
                    values.append(0.)

            return indexes, values
        else:
            return None

    def transform(self, inputs: list):
        """
        convert input of features into feature vectors
        :param inputs: list, list of input feature dict, created by PosToken Class
        :return: list of feature vectors shape (number of inputs, number of features in list)
        """

        n_inputs = len(inputs)
        n_features = 300
        transformed = np.zeros((n_inputs, n_features+1))    # adding extra field with the features for bias
        for i in range(n_inputs):
            features = inputs[i]
            transformed[i][0] = 1   # adding the bias value
            for j in range(len(features)):
                transformed[i][j] = features[j]


            return transformed
        else:
            print("Features list not Found, Please Run fit first")
            return None
