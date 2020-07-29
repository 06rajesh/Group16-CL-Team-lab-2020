import numpy as np
import os.path as path
from scipy import spatial


class Glove:
    """
    Class to handle the glove Embeddings, load the pretrained
    embeddings from the directory and return embedding
    vector for words.
    """
    def __init__(self):
        self.embeddings_dict = {}
        self.n_dimension = 0
        self.average = None

    def load(self):
        """
        Load the embedding vector from directory
        and store the whole embedding as a dictionary
        :return: none
        """
        with open(path.dirname(path.abspath(__file__)) + "pretrained/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

        first_key = next(iter(self.embeddings_dict))
        self.n_dimension = len(self.embeddings_dict[first_key])
        self.calculate_average_vec()

    def calculate_average_vec(self):
        n_vec = len(self.embeddings_dict)
        vecs = np.zeros((n_vec, self.n_dimension), dtype=np.float32)
        for i, key in enumerate(self.embeddings_dict):
            vecs[i] = self.embeddings_dict[key]

        self.average = np.mean(vecs, axis=0)

    def find_closest_words(self, term, limit=5):
        embedding = self.embeddings_dict[term]
        all_words = sorted(self.embeddings_dict.keys(),
                      key=lambda word: spatial.distance.euclidean(self.embeddings_dict[word], embedding))
        return all_words[:limit]

    def get_embedding_val(self, term, average_on_none=False):
        """
        Get the vector of each term
        :param term: a string, for which it will search the dictonary for embedding vector
        :param average_on_none: Boolean, if true will return average vector if not found on Dictionary else
        return None
        :return: Vector or None
        """
        try:
            val = self.embeddings_dict[term.lower()]
        except KeyError:
            if average_on_none:
                val = self.average
            else:
                val = None
        return val
