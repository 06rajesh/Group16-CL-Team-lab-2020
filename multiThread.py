import multiprocessing
import time


def spawn(num, return_dict):
    name = multiprocessing.current_process().name
    print("{} is starting".format(name))
    time.sleep(2)
    return_dict[name] = num*num
    print("{} is exiting".format(name))


def split(a, n):
    k, m = divmod(len(a), n)
    gen = (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    full = list()
    for i in range(len(a)):
        temp = list()
        for j in range(n):
            temp.append(i)

    return list(gen)


def chunkify(items, chunk_len):
    return [items[i:i+chunk_len] for i in range(0,len(items),chunk_len)]


class MultiProcessTest:
    def __init__(self):
        self.count = 0

    def run(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = list()

        pos = ['NFP', 'WRB', 'VB', 'VBG', '-RRB-', 'VBN', 'VBD', ':', 'UH', 'NN', 'JJR', 'WDT', 'DT', 'IN', 'WP$', 'NNPS', 'MD', 'RBR', 'HYPH', 'CC', '``', "''", ',']
        splitted = chunkify(pos, 3)

        print(splitted)

        for batch in splitted:
            for i in batch:
                p = multiprocessing.Process(name="Process_" + str(i), target=spawn, args=(i, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

        print(return_dict)


if __name__ == '__main__':
    mu = MultiProcessTest()
    mu.run()

