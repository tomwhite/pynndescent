import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results/results_beefy_20190227.csv')

for algorithm, rows in (('pynndescent_regular', 50000), ('pynndescent_regular', 100000), ('pynndescent_threaded', 50000), ('pynndescent_threaded', 100000)):
    results_subset = results[(results['algorithm'] == algorithm) & (results['rows'] == rows)][['threads','duration']]
    plt.loglog('threads', 'duration', data=results_subset, marker='o', label='{} {} rows'.format(algorithm, rows))

plt.title("Nearest neighbor algorithms (D=128, NN=25)")
plt.xlabel('Threads')
plt.ylabel('Duration (s)')
plt.legend()

plt.show()