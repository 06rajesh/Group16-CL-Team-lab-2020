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

