from dataProvider import DataProvider
from evaluation import Evaluation
from posToken import PosToken
from multiClassPerceptron import MultiClassItem, MultiClassPerceptron


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
    x_test, y_test, classes = dt.load_test_data()

    # sentences, sentence_pos, classes = dt.load_train_data()
    # items = prepare_multi_class_item(sentences, sentence_pos, classes)
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
    {'precision': 0.10618033858730584, 'recall': 0.09774621372178947, 'fscore': 0.10178886418748978}
    Micor Score: 
    {'precision': 0.22776990393301963, 'recall': 0.22776990393301963, 'fscore': 0.22776990393301963}
    """
