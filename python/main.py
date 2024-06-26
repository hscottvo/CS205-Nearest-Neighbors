import logging
import sys
from argparse import Namespace
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

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
    chunk_size = 100
    assert result.shape[0] % chunk_size == 0

    num_samples = result.shape[0]
    assert result.shape[0] == result.shape[1]

    acc_array = np.empty((num_samples))

    def call_query(sample_idx: int):
        pred_label = query(result, labels, sample_idx)
        acc_array[sample_idx] = pred_label == labels[sample_idx]

    def process_chunk(chunk):
        for idx in chunk:
            call_query(idx)

    chunks = [
        range(i * chunk_size, i * chunk_size + chunk_size)
        for i in range(num_samples // chunk_size)
    ]

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(process_chunk)(i) for i in tqdm(chunks)
    )

    assert np.all(
        (acc_array == 0) | (acc_array == 1)
    ), "Did not predict for all samples"

    accuracy = acc_array.sum() / num_samples
    return accuracy


def config_logger(args: Namespace):
    logger = logging.getLogger(__name__)
    print_handler = logging.StreamHandler(sys.stdout)

    if args.verbose:
        print_handler.setLevel(logging.INFO)
        logger.addHandler(print_handler)

    logging.basicConfig(filename=args.log.name, encoding="utf-8", level=logging.DEBUG)

    return logger


if __name__ == "__main__":
    args = parse_main()

    logger = config_logger(args)

    logger.info(f"Running at {datetime.now()}")

    timer_start = datetime.now()
    logger.info(f"Reading from {args.file.name} at {timer_start}")
    features, labels = read(args.file)
    timer_end = datetime.now()
    logger.info(f"Took {(timer_end - timer_start).total_seconds()} seconds to read")

    logger.info("Starting search")

    feature_set = set()
    prev_acc = 0
    while True:
        print(f"Trying with {len(feature_set) + 1} features")
        timer_start = datetime.now()
        best_acc = prev_acc
        best_idx = -1
        for i in range(features.shape[2]):
            if i not in feature_set:
                features_to_try = list(feature_set) + [i]
                logger.info(f"\t\tTrying features {features_to_try}")
                acc = n_fold_accuracy(features[:, :, features_to_try], labels)
                logger.info(f"\t\t\tAccuracy: {acc}")
                if acc > best_acc:
                    logger.info(f"\t\t\t{acc} beats {best_acc}")
                    best_acc = acc
                    best_idx = i
        timer_end = datetime.now()
        logger.info(
            f"\tTook {(timer_end - timer_start).total_seconds()} seconds to run with {len(feature_set) + 1} features"
        )
        if best_idx == -1:
            logger.info(
                f"Finished search with {feature_set} as final features and accuracy {best_acc}"
            )
            break
        else:
            prev_acc = best_acc
            feature_set.add(best_idx)
