from dataProvider import DataProvider
from evaluation import Evaluation

if __name__ == '__main__':
    dt = DataProvider(path='data')
    dt.load_data()

    ev = Evaluation(dt.tagset, dt.original_label, dt.predicted_label)
    ev.calculate()

    print("Micro")
    print(ev.micro)

    print("Macro")
    print(ev.macro)

    # Output
    ############################
    # Micro
    # {'precision': 0.9436296630565584, 'recall': 0.9435232186501222, 'fscore': 0.9435764378513547}
    # Macro
    # {'precision': 0.8512286029127887, 'recall': 0.8280785657646288, 'fscore': 0.8394940172773762}
