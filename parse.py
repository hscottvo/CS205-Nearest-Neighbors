from argparse import ArgumentParser, FileType
from typing import Tuple

import numpy as np


def read_file(file) -> Tuple[np.ndarray, np.ndarray]:
    lines = [i.split() for i in file.read().split("\n")[:-1]]
    arr = np.array(lines)
    labels = arr[:, 0].astype(np.float64).astype(np.int8)
    features = arr[:, 1:].astype(np.float64)
    return labels, features


def parse_preprocess():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=FileType("r"),
        required=True,
        help="The data file to pass in",
    )
    return parser.parse_args()


def parse_main():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=FileType("r"),
        required=True,
        help="The data file that features/labels are based on ",
    )
    return parser.parse_args()
