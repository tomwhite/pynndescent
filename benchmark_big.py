# Run on a GCE instance
# n1-highmem-96 - 624GB mem, 100GB disk
# gcloud compute --project=hca-scale instances create ll-knn-big --zone=us-east1-b --machine-type=n1-highmem-96 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=218219996328-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --min-cpu-platform="Intel Skylake" --image=ubuntu-1804-bionic-v20190429 --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-standard --boot-disk-device-name=ll-knn
# gcloud compute --project "hca-scale" ssh --zone "us-east1-b" "ll-knn-big"
#
# sudo apt-get update && sudo apt-get install -y git python3-pip
# pip3 install numba numpy scipy scikit-learn
# pip3 install git+https://github.com/tomwhite/pynndescent@benchmarks
# pip3 list
# git clone https://github.com/tomwhite/pynndescent
# cd pynndescent
# git checkout benchmarks
# python3 benchmark_big.py

import multiprocessing
import time

import numpy as np

from pynndescent import NNDescent

np.random.seed(42)

N = int(1e7)
D = 30
dataset = np.random.rand(N, D).astype(np.float32)

n_cores = multiprocessing.cpu_count()

def pynndescent_threaded(X, threads, n_neighbors, max_candidates):
    t0 = time.time()
    index = NNDescent(X, n_neighbors=n_neighbors, max_candidates=max_candidates, n_jobs=threads, verbose=True)
    indices, distances = index._neighbor_graph
    t1 = time.time()
    return indices, distances, t1-t0

def accuracy(expected, actual):
    # Look at the size of corresponding row intersections
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(expected, actual)]).sum() / expected.size

def all_experiments():
    n_neighbors = D # make K=D for good accuracy, see Dong paper, section 4.5
    max_candidates = 50
    threads = n_cores
    for rows in (1e6, 2e6, 5e6, 1e7):
        yield (pynndescent_threaded, int(rows), threads, n_neighbors, max_candidates)

def generate_experiments(predicate=None):
    for exp in all_experiments():
        if predicate is None or predicate(exp):
            yield exp

if __name__ == '__main__':
    # modify the predicate to run a subset of experiments
    predicate = None
    for algorithm, rows, threads, n_neighbors, max_candidates in generate_experiments(predicate):
        subset = dataset[:rows]
        indices, distances, t = algorithm(subset, threads, n_neighbors, max_candidates)
        print("{},{},{},{},{},{},{}".format(algorithm.__name__, threads, rows, n_neighbors, max_candidates, t, -1))

        # run a second time to get an estimate of relative accuracy
        indices_check, distances_check, t_check = algorithm(subset, threads, n_neighbors, max_candidates)
        est_acc = accuracy(indices, indices_check)
        print("{},{},{},{},{},{},{}".format(algorithm.__name__, threads, rows, n_neighbors, max_candidates, t_check, est_acc))