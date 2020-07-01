import embeddings
import data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rnn import PosRNN


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


if __name__ == '__main__':
    # g = embeddings.Glove()
    # g.load()
    # closest = g.find_closest_words("king")

    d = data.Provider()
    sentences, sentence_pos, classes = d.load_train_data()

    model = PosRNN()
    model.train(sentences, sentence_pos, classes)

    test_sentences, test_sentence_pos, _ = d.load_test_data()
    model.evaluate(test_sentences, test_sentence_pos)
    # plot_embeddings_by_class([nn_list, jj_list])
