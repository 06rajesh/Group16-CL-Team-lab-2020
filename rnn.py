import os
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import backend as K

import embeddings


class PosRNN:
    def __init__(self):
        self.words_to_index = dict()
        self.tags_to_index = dict()
        self.max_length = 0
        self.vocab_size = 0
        self.feature_length = 0
        self.embedding_matrix = None
        self.model = None

    def save_props_to_file(self):
        """
        Save weight to _savePath directory in weights.pickle file
        """
        filename = os.path.join("checkpoints", "props.pickle")
        print("Saving Properties to file {}".format(filename))
        print("===================================")
        outfile = open(filename, 'wb')
        pickle.dump([self.words_to_index, self.tags_to_index, self.max_length], outfile)
        outfile.close()

    def load_properties(self):
        filename = os.path.join("checkpoints", "props.pickle")
        if len(self.words_to_index) == 0 and os.path.isfile(filename):
            print("Loading Properties from file {}".format(filename))
            print("===================================")
            infile = open(filename, 'rb')
            w2i, t2i, max_len, voc_size = pickle.load(infile)
            infile.close()
            self.words_to_index = w2i
            self.tags_to_index = t2i
            self.max_length = max_len

    def set_training_props(self, train_x, tagset):
        words, tags = set([]), set([])
        max_count = 0
        for s in train_x:
            word_count = 0
            for w in s:
                words.add(w.lower())
                word_count += 1
            if word_count > max_count:
                max_count = word_count

        words = list(words)
        self.max_length = max_count
        # self.vocab_size = len(words)
        self.create_embedded_matrix(words)

        self.words_to_index = {w: i + 2 for i, w in enumerate(words)}
        self.words_to_index['-PAD-'] = 0  # The special value used for padding
        self.words_to_index['-OOV-'] = 1  # The special value used for OOVs

        self.tags_to_index = {t: i + 1 for i, t in enumerate(tagset)}
        self.tags_to_index['-PAD-'] = 0  # The special value used to padding

        self.save_props_to_file()

    def encode_sentences(self, sentences):
        encoded = list()
        for s in sentences:
            s_int = []
            for w in s:
                try:
                    s_int.append(self.words_to_index[w.lower()])
                except KeyError:
                    s_int.append(self.words_to_index['-OOV-'])

            encoded.append(s_int)

        encoded = sequence.pad_sequences(encoded, maxlen=self.max_length, padding='post')
        return encoded

    def encode_tags(self, sentence_tags):
        encoded = list()
        for s in sentence_tags:
            tag_int = []
            for tag in s:
                tag_int.append(self.tags_to_index[tag])

            encoded.append(tag_int)

        encoded = sequence.pad_sequences(encoded, maxlen=self.max_length, padding='post')
        return encoded

    def create_embedded_matrix(self, words):
        embed = embeddings.Glove()
        embed.load()
        vocab_size = len(embed.embeddings_dict)

        e_matrix = np.zeros((vocab_size, embed.n_dimension))
        i = 0
        for word in words:
            e_matrix[i] = embed.get_embedding_val(word)
            i += 1

        self.embedding_matrix = e_matrix
        self.feature_length = embed.n_dimension
        self.vocab_size = vocab_size

    @staticmethod
    def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    @staticmethod
    def ignore_class_accuracy(to_ignore=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)

            ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy

        return ignore_accuracy

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_length,)))
        model.add(layers.Embedding(self.vocab_size, self.feature_length, weights=[self.embedding_matrix],
                                   input_length=self.max_length, trainable=False))
        model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
        model.add(layers.TimeDistributed(layers.Dense(len(self.tags_to_index))))
        model.add(layers.Activation('relu'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy', self.ignore_class_accuracy(0)])

        model.summary()
        return model

    def train(self, sentences, sentence_pos, tagset):
        self.set_training_props(sentences, tagset)
        train_x = self.encode_sentences(sentences)
        train_y = self.encode_tags(sentence_pos)

        cat_train_tags_y = self.to_categorical(train_y, len(self.tags_to_index))

        model = self.create_model()
        model.fit(train_x, cat_train_tags_y, batch_size=128, epochs=40, validation_split=0.2)

        # serialize weights to HDF5
        filename = os.path.join("checkpoints", "model.h5")
        model.save_weights(filepath=filename)
        print("Saved model to disk")
        print("===================================")
        self.model = model
        return self

    # def predict(self, sentences):
    #     self.load_properties()

    def evaluate(self, sentences, sentence_pos):
        test_x = self.encode_sentences(sentences)
        test_y = self.encode_tags(sentence_pos)

        scores = self.model.evaluate(test_x, self.to_categorical(test_y, len(self.tags_to_index)))
        print("=============================================")  # acc: 99.09751977804825
        print("Final Accuracy: " + str(scores[1] * 100))

