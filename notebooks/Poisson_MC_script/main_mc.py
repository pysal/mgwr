from multiprocessing import Process, Queue
import multiprocessing as mp


import f_2
import warnings
import pickle
import sys
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    for i in range(int(sys.argv[1]),int(sys.argv[2]), 50):
        start = i
        end = i + 50

        print("Starting iterations for indexes: {} to {}".format(str(start), str(end)))

        pool = mp.Pool(processes=50)
        result = pool.map(f_2.models, range(start,end))

        print("Completing iterations for indexes: {} to {}".format(str(start), str(end)))
        with open("pkls/results-{}-{}.pkl".format(str(start), str(end)), 'wb') as out:
           pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)
