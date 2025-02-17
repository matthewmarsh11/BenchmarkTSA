import torch
from models import *
from utils import *
import os


import matplotlib.pyplot as plt

def is_pareto_efficient(costs):
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index + 1])
    return is_efficient

model_folder = '/Users/MatthewMarsh/Desktop/Academia/Imperial College London/PhD Research/BenchmarkTSA/models'
model_files = [f for f in os.listdir(model_folder) if f.startswith('CSTR_NLL')]

means = []
vars = []
labels = []

for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    model = torch.load(model_path)
    means.append(model['mse'])
    vars.append(model['var'])
    labels.append(model_file)

means = np.array(means)
vars = np.array(vars)

plt.scatter(means, vars)

for i, label in enumerate(labels):
    plt.annotate(label, (means[i], vars[i]))

costs = np.vstack((means, vars)).T
pareto_efficient_indices = is_pareto_efficient(costs)
pareto_means = means[pareto_efficient_indices]
pareto_vars = vars[pareto_efficient_indices]

plt.scatter(pareto_means, pareto_vars, color='red')
plt.xlabel('Mean Squared Error')
plt.ylabel('Variance')
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
plt.title('Mean vs Variance Scatter Plot with Pareto Frontier')
plt.show()