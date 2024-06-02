import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from parse import parse_main
from preprocess import read


def query(diffs: np.ndarray, labels: np.ndarray, point: int) -> int:
    assert (
        point >= 0 and point < diffs.shape[0]
    ), f"Sample {point} is out of range for data of size {diffs.shape[0]}"
    assert diffs.ndim == 3, "Must be a nxnxm array"

    sums = diffs.sum(axis=2)

    before = sums[0:point, point]
    mid_place = np.array([np.infty])
    after = sums[point, point + 1 :]
    dists_squared = np.concatenate([before, mid_place, after])
    min_dist_idx = np.argmin(dists_squared)

    return labels[min_dist_idx]


def n_fold_accuracy(result: np.ndarray, labels: np.ndarray) -> float:
    num_samples = result.shape[0]
    assert result.shape[0] == result.shape[1]

    acc_array = np.empty((num_samples))
    for i in range(num_samples):
        pred_label = query(result, labels, i)
        acc_array[i] = pred_label == labels[i]

    assert np.all(
        (acc_array == 0) | (acc_array == 1)
    ), "Did not predict for all samples"

    accuracy = acc_array.sum() / num_samples
    return accuracy


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logging.basicConfig(filename="run.log", encoding="utf-8", level=logging.DEBUG)
    logger.info(f"Running at {datetime.now()}")

    args = parse_main()

    logger.info(f"Reading from {args.file.name}")
    features, labels = read(args.file)

    logger.info("Starting search")

    feature_set = set()
    prev_acc = 0
    while True:
        best_acc = prev_acc
        best_idx = -1
        for i in range(features.shape[2]):
            if i not in feature_set:
                features_to_try = list(feature_set) + [i]
                logger.info(f"\tTrying features {features_to_try}")
                acc = n_fold_accuracy(features[:, :, features_to_try], labels)
                logger.info(f"\t\tAccuracy: {acc}")
                if acc > best_acc:
                    logger.info(f"\t\t{acc} beats {best_acc}")
                    best_acc = acc
                    best_idx = i
        if best_idx == -1:
            logger.info(
                f"Finished search with {feature_set} as final features and accuracy {best_acc}"
            )
            break
        else:
            prev_acc = best_acc
            feature_set.add(best_idx)
