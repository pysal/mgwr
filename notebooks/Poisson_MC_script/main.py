from multiprocessing import Process, Queue


import f
import warnings
import pickle
import sys
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    for i in range(int(sys.argv[1]),int(sys.argv[2]), 10):
        start = i
        end = i + 10
        # Define an output queue
        output=Queue()
        print("Starting iterations for indexes: {} to {}".format(str(start), str(end)))
        # Setup a list of processes that we want to run
        processes = [Process(target=f.models, args=(output,)) for x in range(start, end)]
        for p in processes:
        	# Run process
        	p.start()

        for p in processes:
        	# Exit the completed process
        	p.join()

        # Get process results from the output queue
        result = [output.get(p) for p in processes]
        print("Completing iterations for indexes: {} to {}".format(str(start), str(end)))
        with open("pkls/result-{}-{}.pkl".format(str(start), str(end)), 'wb') as out:
           pickle.dump(result, out, pickle.HIGHEST_PROTOCOL)
