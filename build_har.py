# -*- coding: utf-8 -*-
"""Build artifacts/data/har_data.npz, the one input no other script creates.

The file holds the UCI "Human Activity Recognition Using Smartphones" data:
the 561 pre-computed features, train and test part merged in that order, and
the activity labels shifted to 0..5:

    X  float64, (10299, 561)
    y  int64,   (10299,)

It is fetched from OpenML (dataset "har", version 1, OpenML id 1478), which
mirrors the UCI release with the feature values rounded to six decimals. That
copy is what the campaign was run on: the file written here is byte-identical
to the one behind the reported numbers, and the script checks that against
DATA_CHECKSUMS.sha256. Reading the raw UCI text files instead gives the same
rows to seven significant digits but a different checksum.

Usage (repo root):
    python build_har.py
The download (about 60 MB) is cached by scikit-learn under artifacts/data/openml.
"""
import hashlib
import os

import numpy as np
from sklearn.datasets import fetch_openml

from wpaths import DATA, HERE

OUT = os.path.join(DATA, "har_data.npz")
CHECKSUMS = os.path.join(HERE, "DATA_CHECKSUMS.sha256")


def expected_sha():
    if not os.path.exists(CHECKSUMS):
        return None
    for line in open(CHECKSUMS, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and parts[1].replace("\\", "/").endswith("har_data.npz"):
            return parts[0]
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = fetch_openml("har", version=1, as_frame=True,
                     data_home=os.path.join(DATA, "openml"))
    X = d.data.to_numpy().astype(np.float64)      # column-major, as pandas hands it over
    y = d.target.to_numpy().astype(np.int64) - 1  # OpenML labels 1..6 -> 0..5
    assert X.shape == (10299, 561) and y.shape == (10299,), (X.shape, y.shape)
    assert y.min() == 0 and y.max() == 5

    np.savez(OUT, X=X, y=y)
    got, want = sha256(OUT), expected_sha()
    print("written", OUT, X.shape, y.shape)
    print("sha256 ", got)
    if want is None:
        print("no reference checksum in", CHECKSUMS)
    elif got == want:
        print("matches DATA_CHECKSUMS.sha256")
    else:
        print("DIFFERS from DATA_CHECKSUMS.sha256:", want)


if __name__ == "__main__":
    main()
