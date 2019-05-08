import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('results/convergence.csv')

print(results)

plt.semilogy('iteration', 'c', data=results)

#
# for algorithm, threads in (('scikitlearn_brute', 1), ('scikitlearn_ball_tree', 1), ('pynndescent_regular', 8), ('pynndescent_threaded', 1), ('pynndescent_threaded', 8)):
#     results_subset = results[(results['algorithm'] == algorithm) & (results['threads'] == threads)][['rows','duration']]
#     plt.loglog('rows', 'duration', data=results_subset, marker='o', label='{} (threads={})'.format(algorithm, threads))
#
# plt.title("Nearest neighbor algorithms (D=128, NN=25)")
# plt.xlabel('Rows')
# plt.ylabel('Duration (s)')
# plt.legend()
#
plt.show()