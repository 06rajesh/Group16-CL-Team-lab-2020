from dataProvider import DataProvider
from evaluation import Evaluation
from posToken import PosToken
from multiClassPerceptron import MultiClassItem, MultiClassPerceptron
from dictVectorizer import CustomDictVectorizer

savePath = "weights"


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
    X = list()
    t = PosToken()
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            features = t.get_features(sentences[i], j)
            X.append(features)
            for k, v in inputs.items():
                item = inputs.get(k)
                if sentence_pos[i][j] == k:
                    item.Y.append(1.)
                else:
                    item.Y.append(0.)

    dv = CustomDictVectorizer(save_to=savePath)
    dv.fit(X, min_occurs=300)
    for k, v in inputs.items():
        item = inputs.get(k)
        item.X = X

    return inputs


def prepare_training_data(s, s_p):
    x = list()
    y = list()
    t = PosToken()
    for i in range(len(s)):
        for j in range(len(s[i])):
            features = t.get_features(s[i], j)
            x.append(features)
            y.append(s_p[i][j])

    dv = CustomDictVectorizer(save_to=savePath)
    x_transformed = dv.transform(x)
    return x_transformed, y


if __name__ == '__main__':
    dt = DataProvider(path='data')

    sentences, sentence_pos, classes = dt.load_train_data()
    sentences_test, sentence_pos_test, _ = dt.load_test_data()

    items = prepare_multi_class_item(sentences, sentence_pos, classes)
    x_test, y_test = prepare_training_data(sentences_test, sentence_pos_test)

    mlp = MultiClassPerceptron(save_to=savePath, n_process=2)
    mlp.train(items)
    y_pred = mlp.predict(inputs=x_test)

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
    {'precision': 0.7095003745834969, 'recall': 0.6112155768980928, 'fscore': 0.656700905707975}
    Micro Score: 
    {'precision': 0.7788986969314838, 'recall': 0.7788986969314838, 'fscore': 0.7788986969314838}
    """
