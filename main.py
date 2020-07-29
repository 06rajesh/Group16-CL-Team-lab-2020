import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rnn import PosRNN
from evaluation import Evaluation
import perceptron
import embeddings
import data


def plot_embeddings_by_class(words_by_class):
    joined = list()
    labels = list()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    shapes = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p"]

    for i in range(len(words_by_class)):
        single_class_words = list(words_by_class[i])
        joined.extend(single_class_words)
        labels.extend([i]*len(single_class_words))

    vectors = [g.get_embedding_val(word) for word in joined]

    tsne = TSNE(n_components=3, random_state=0)
    Y = tsne.fit_transform(vectors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(Y)):
        ax.scatter(Y[i][0], Y[i][1], Y[i][2], c=colors[labels[i]], marker=shapes[labels[i]])
        ax.text(Y[i][0], Y[i][1], Y[i][2], '%s' % (joined[i]), size=5, zorder=1, color=colors[labels[i]])

    plt.show()


def check_embedding_stats(type="GLOVE"):
    d = data.Provider()
    sentences, sentence_pos, classes = d.load_train_data()

    if type == "GLOVE":
        embed = embeddings.Glove()
    else:
        embed = embeddings.FastText()
    embed.load()

    not_found_count = 0
    not_found_words = list()
    for sentence in sentences:
        for word in sentence:
            val = embed.get_embedding_val(word)
            if val is None:
                not_found_count += 1
                not_found_words.append(word)

    not_found_words = set(not_found_words)
    print(len(not_found_words))
    print("Total Not Found {}".format(not_found_count))
    print("Total Terms in Embedding: {}".format(len(embed.embeddings_dict)))


def run_rnn_model(embeddings='GLOVE', train=False):
    d = data.Provider()

    if embeddings == 'FASTTEXT':
        model = PosRNN(save_path="rnn_fasttext_model", embedding_type=embeddings)
    else:
        model = PosRNN(save_path="rnn_glove_model", embedding_type=embeddings)

    if train:
        sentences, sentence_pos, classes = d.load_train_data()
        print("Training Started using {} embeddings".format(embeddings))
        print("===================================================")
        model.train(sentences, sentence_pos, classes, num_of_epochs=15)

    test_sentences, test_sentence_pos, classes = d.load_test_data()
    model.evaluate(test_sentences, test_sentence_pos)
    predictions = model.predict(test_sentences)

    ev = Evaluation()
    ev.fit(original=test_sentence_pos, predicted=predictions, classes=classes)

    print("Macro Score: ")
    macro = ev.get_macro_score()
    print(macro)

    print("Micro Score: ")
    micro = ev.get_micro_score()
    print(micro)

    print("Accuracy: ")
    accuracy = ev.get_accuracy()
    print(accuracy)


def run_perceptron_with_embeddings(train=False):
    d = data.Provider()

    mlp = perceptron.MultiClassPerceptron(n_process=4)

    classes = None
    if train:
        sentences, sentence_pos, classes = d.load_train_data()
        mlp.fit(sentences, sentence_pos, classes)
        mlp.train()

    sentences_test, sentence_pos_test, test_classes = d.load_test_data()
    y_pred = mlp.predict(sentences=sentences_test)

    if classes is None:
        classes = test_classes

    ev = Evaluation()
    ev.fit(original=sentence_pos_test, predicted=y_pred, classes=classes)

    print("Macro Score: ")
    macro = ev.get_macro_score()
    print(macro)

    print("Micro Score: ")
    micro = ev.get_micro_score()
    print(micro)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # check_embedding_stats(type="GLOVE")
    # check_embedding_stats(type="FASTTEXT")

    # Define the program description
    text = 'Program to run different models for parts-of-speech tagging'

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--model", "-m", help="Select model, perceptron/rnn", default='rnn')
    parser.add_argument("--embeddings", "-e", help="Select embeddings for RNN, glove/fasttext", default='glove')
    parser.add_argument("--train", "-t", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate Training mode.")

    args = parser.parse_args()

    if args.model.lower() == 'perceptron':
        run_perceptron_with_embeddings(train=args.train)
    else:
        embed = "GLOVE"
        if args.embeddings.lower() == 'fasttext':
            embed = 'FASTTEXT'
        run_rnn_model(embeddings=embed, train=args.train)

