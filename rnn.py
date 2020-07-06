import os
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras import backend as K

import embeddings

# https://medium.com/swlh/named-entity-recognition-ner-using-keras-bidirectional-lstm-28cd3f301f54
class PosRNN:
    def __init__(self):
        self.words_to_index = dict()
        self.tags_to_index = dict()
        self.max_length = 0
        self.vocab_size = 0
        self.feature_length = 0
        self.embedding_matrix = None
        self._model = None

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
            w2i, t2i, max_len = pickle.load(infile)
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
            val = embed.get_embedding_val(word)
            if val is not None:
                e_matrix[i] = embed.get_embedding_val(word)
                i += 1

        self.embedding_matrix = e_matrix
        self.feature_length = embed.n_dimension
        self.vocab_size = vocab_size

        filename = os.path.join("checkpoints", "embeddings.pickle")
        print("Saving Embeddings Matrix to file {}".format(filename))
        print("===================================")
        outfile = open(filename, 'wb')
        pickle.dump(self.embedding_matrix, outfile)
        outfile.close()

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

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.max_length,)))
        model.add(layers.Embedding(self.vocab_size, self.feature_length, weights=[self.embedding_matrix],
                                   input_length=self.max_length, trainable=False))
        model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
        model.add(layers.TimeDistributed(layers.Dense(len(self.tags_to_index))))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.summary()
        return model

    def train(self, sentences, sentence_pos, tagset):
        self.set_training_props(sentences, tagset)
        train_x = self.encode_sentences(sentences)
        train_y = self.encode_tags(sentence_pos)

        cat_train_tags_y = self.to_categorical(train_y, len(self.tags_to_index))

        model = self.create_model()
        model.fit(train_x, cat_train_tags_y, batch_size=128, epochs=20, validation_split=0.2)

        # serialize weights to HDF5
        filename = os.path.join("checkpoints", "weights.h5")
        model.save_weights(filepath=filename)
        print("Saved Weights to disk")
        print("===================================")

        filename = os.path.join("checkpoints", "my_model")
        model.save(filepath=filename)
        print("Saved Model to disk")
        print("===================================")

        self._model = model
        return self

    def get_tag_from_idx(self, id_to_convert):
        tag = None
        for key, idx in self.tags_to_index.items():
            if idx == id_to_convert:
                tag = key
                break
        return tag

    def get_model(self):
        if self._model is None:
            self.load_properties()
            loaded_model = keras.models.load_model('checkpoints/my_model')
            return loaded_model
        else:
            return self._model

    def predict(self, sentences):
        model = self.get_model()
        test_x = self.encode_sentences(sentences)
        pred_values = model.predict(test_x)

        predictions = list()
        for i, sentence in enumerate(sentences):
            sentence_pred = list()
            for j, word in enumerate(sentence):
                prediction = pred_values[i][j]
                max_idx = np.argmax(prediction)
                tag = self.get_tag_from_idx(max_idx)
                sentence_pred.append(tag)
            predictions.append(sentence_pred)

        return predictions

    def evaluate(self, sentences, sentence_pos):
        model = self.get_model()

        # Check its architecture
        model.summary()

        test_x = self.encode_sentences(sentences)
        test_y = self.encode_tags(sentence_pos)

        # Evaluate the model
        loss, acc = model.evaluate(test_x, self.to_categorical(test_y, len(self.tags_to_index)))
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

