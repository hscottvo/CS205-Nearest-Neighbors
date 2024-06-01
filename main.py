import numpy as np
from joblib import Parallel
from tqdm import tqdm

import parse


def calc_diff_squared(features: np.ndarray, i: int, j: int, feature: int):
    return (features[i, feature] - features[j, feature]) ** 2


def parallel_dist(features: np.ndarray):
    dist_matrix = np.zeros((features.shape[0], features.shape[0], features.shape[1]))

    pass


args = parse.parse()

labels, features = parse.read_file(args.file)

num_samples = features.shape[0]
num_features = features.shape[1]

dist_matrix = np.zeros((features.shape[0], features.shape[0], features.shape[1]))
# print(dist_matrix.shape)

print(num_samples)
for i in tqdm(range(num_samples)):
    for j in range(num_samples):
        for feature in range(num_features):
            dist_matrix[i, j, feature] = (
                features[i, feature] - features[j, feature]
            ) ** 2
print(dist_matrix[:, :, 0])
