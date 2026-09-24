"""Part 4 -- write the online lookup yourself.

The tables, the grid, and the scaling are done. Fill in the two TODOs,
then check against classify_point.py.
"""

import numpy as np
from pipeline_common import (
    load_split_scaled,
    scale_features,
    quantize,
    dequantize,
    GRID_N,
    REJECT_TAU
)

bounds = load_split_scaled()["bounds"]

surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)

# The shipped model
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def classify(raw_point):
    """Class label for a RAW point, or None for "unknown"."""

    # Scale the raw point using the training bounds
    point, _ = scale_features(np.asarray(raw_point), bounds)

    # Find the grid cell
    col = int(point[0] * GRID_N)
    row = int(point[1] * GRID_N)

    # Keep the indexes inside the valid grid
    col = int(np.clip(col, 0, GRID_N - 1))
    row = int(np.clip(row, 0, GRID_N - 1))

    # Read and dequantize one stored value for each class
    values = np.array([
        dequantize(tables[c][row, col])
        for c in classes
    ])

    # Find the class with the largest value
    best = int(np.argmax(values))

    # Reject if even the strongest class is too weak
    if values[best] < REJECT_TAU:
        return None

    return classes[best]


if __name__ == "__main__":

    # One raw point on each shape, then the ring's hollow centre
    for q in (
        [0.0, -1.8],
        [0.0, 5.0],
        [6.8, 1.6],
        [4.8, 1.6]
    ):
        print(q, "->", classify(q))