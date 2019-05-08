import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results/results_big_20190508.csv')

for algorithm, threads in (('pynndescent_threaded', 96),):
    results_subset = results[(results['algorithm'] == algorithm) & (results['threads'] == threads)][['rows','duration']]
    plt.loglog('rows', 'duration', data=results_subset, marker='o', label='{} (threads={})'.format(algorithm, threads))

plt.title("Nearest neighbor algorithms (D=50, NN=30)")
plt.xlabel('Rows')
plt.ylabel('Duration (s)')
plt.legend()

plt.show()