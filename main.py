from dataProvider import DataProvider
from evaluation import Evaluation
from posToken import PosToken
from multiClassPerceptron import MultiClassItem, MultiClassPerceptron


def prepare_multi_class_item(sentences, sentence_pos, classes):
    inputs = dict()
    for pos in classes:
        inputs[pos] = MultiClassItem(pos)

    for i in range(0, len(sentences)):
        for j in range(0, len(sentences[i])):
            token = PosToken(sentences[i][j])
            features = token.get_features()

            for k, v in inputs.items():
                item = inputs.get(k)
                if sentence_pos[i][j] == k:
                    item.X.append(features)
                    item.Y.append(1.)
                else:
                    item.X.append(features)
                    item.Y.append(0.)

    return inputs


if __name__ == '__main__':
    dt = DataProvider(path='data')
    # original, _ = dt.load_original_data()
    # predicted, _ = dt.load_predicted_data()

    sentences, sentence_pos, classes = dt.load_train_data()
    items = prepare_multi_class_item(sentences, sentence_pos, classes)
    mlp = MultiClassPerceptron()
    mlp.train(items)
    mlp.predict('stocks')
