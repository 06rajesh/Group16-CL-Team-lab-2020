from dataProvider import DataProvider
from evaluation import Evaluation
from posToken import PosToken
from multiClassPerceptron import MultiClassItem, MultiClassPerceptron
from perceptron import Perceptron


def prepare_multi_class_item(sentences, sentence_pos, classes):
    """
    Preapre Items for MultiClassPerceptron Using MultiClassItem
    :param sentences: List of Sentences, which each is a list of tokens
    :param sentence_pos: List of Sentenece Pos, which each is list of POS tag
    :param classes: list of classes in data
    :return: list of MultiClassItem
    """
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
    x_test, y_test, _ = dt.load_test_data()

    sentences, sentence_pos, classes = dt.load_train_data()

    # for i in range(len(sentences[0])):
    #     print("{} : {}".format(sentences[0][i], sentence_pos[0][i]))
    #     token = PosToken(sentences[0][i])
    #     features = token.get_features()
    #     print(features)
    #     print(pcp.convert_to_feature_vector(features))

    items = prepare_multi_class_item(sentences, sentence_pos, classes)
    mlp = MultiClassPerceptron()
    # mlp.train(items)
    y_pred = mlp.predict(x=x_test)

    ev = Evaluation(original=y_test, predicted=y_pred, classes=classes)
    ev.calculate()

    print("Macro Score: ")
    macro = ev.get_macro_score()
    print(macro)

    print("Micro Score: ")
    micro = ev.get_micro_score()
    print(micro)

    """
    OUTPUT
    ================
    Macro Score: 
    {'precision': 0.4688312691886069, 'recall': 0.41199408085521244, 'fscore': 0.438578902879972}
    Micro Score: 
    {'precision': 0.5829115853033511, 'recall': 0.5829115853033511, 'fscore': 0.5829115853033511}
    """
