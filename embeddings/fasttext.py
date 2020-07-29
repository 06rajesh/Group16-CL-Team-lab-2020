import numpy as np
import os.path as path
import io
from scipy import spatial


class FastText:
    """
    Class to handle the FastText Embeddings, load the pretrained
    embeddings from the directory and return embedding
    vector for words.
    """
    def __init__(self):
        self.embeddings_dict = {}
        self.n_dimension = 0
        self.n_tokens = 0
        self.average = None

    def load(self):
        """
        Load the embedding vector from directory
        and store the whole embedding as a dictionary
        :return: none
        """
        fname = path.join(path.dirname(path.abspath(__file__)), 'pretrained/wiki-news-300d-1M.vec')
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        ft_dict = {}
        for line in fin:
            values = line.split()
            token = values[0]
            vector = np.asarray(values[1:], "float32")
            ft_dict[token] = vector

        self.embeddings_dict = ft_dict
        self.n_tokens = n
        self.n_dimension = d
        self.calculate_average_vec()

    def calculate_average_vec(self):
        n_vec = self.n_tokens
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
            val = self.embeddings_dict[term]
        except KeyError:
            if average_on_none:
                val = self.average
            else:
                val = None
        return val
