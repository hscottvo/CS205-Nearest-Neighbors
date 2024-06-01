import os
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

import parse


# https://stackoverflow.com/questions/36423975/how-to-use-nested-loops-in-joblib-library-in-python
# https://github.com/joblib/joblib/issues/1103
def parallel_diff_squared(features: np.ndarray):
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
    return dist_matrix


def diff_squared(features: np.ndarray):
    num_samples = features.shape[0]
    num_features = features.shape[1]
    dist_matrix = np.zeros((num_samples, num_samples, num_features))
    for i in tqdm(range(num_samples)):
        for j in range(i):
            for k in range(num_features):
                dist_matrix[j, i, k] = (features[i, k] - features[j, k]) ** 2
    return dist_matrix


def write(filename: TextIOWrapper):
    labels, features = parse.read_file(filename)

    path = Path(filename.name)
    features_write_path = path.parent / f"{path.stem}_features.csv"
    labels_write_path = path.parent / f"{path.stem}_labels.csv"

    num_samples = features.shape[0]
    num_features = features.shape[1]

    cache_array = diff_squared(features)
    write_array = cache_array.reshape(num_samples * num_samples, num_features)

    features_write_df = pl.DataFrame(write_array)
    features_write_df.write_csv(features_write_path)

    labels_write_df = pl.DataFrame(labels)
    labels_write_df.write_csv(labels_write_path)

    print("Successfully wrote to")
    print(f"\t{features_write_path} ")
    print("\t\tand")
    print(f"\t{labels_write_path}")


args = parse.parse_preprocess()

write(filename=args.file)
