import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import parse


# https://stackoverflow.com/questions/36423975/how-to-use-nested-loops-in-joblib-library-in-python
# https://github.com/joblib/joblib/issues/1103
def parallel_dist(features: np.ndarray):
    num_samples = features.shape[0]
    num_features = features.shape[1]
    dist_matrix = np.zeros((features.shape[0], features.shape[0], features.shape[1]))

    def calc_diff_squared(i: int, j: int, k: int):
        if i == j:
            return
        val = (features[i, k] - features[j, k]) ** 2
        dist_matrix[i, j, k] = val

    Parallel(n_jobs=4, require="sharedmem")(
        delayed(calc_diff_squared)(i, j, k)
        for i in tqdm(range(num_samples))
        for j in range(i)
        for k in range(num_features)
    )


def dist(features: np.ndarray):
    num_samples = features.shape[0]
    num_features = features.shape[1]
    dist_matrix = np.zeros((num_samples, num_samples, num_features))
    for i in tqdm(range(num_samples)):
        for j in range(i):
            for k in range(num_features):
                dist_matrix[i, j, k] = (features[i, k] - features[j, k]) ** 2


args = parse.parse()

labels, features = parse.read_file(args.file)

# parallel_dist(features)
dist(features)
