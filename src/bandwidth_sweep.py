"""Part 2 tuning -- choose the KDE bandwidth h on the VALIDATION set.

The test set is never touched here. For each candidate h we fit the
per-class KDE on training data, then score the held-out validation points.
The h with the highest validation mean log-likelihood is selected.
"""

import numpy as np

from pipeline_common import load_split_scaled, fit_density


SWEEP = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

data = load_split_scaled()
Xtr = data["train"]
Xval = data["val"]

labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})

print(f"{'h':>6} {'val mean log-likelihood':>26}")

best_h = None
best_ll = -np.inf

for h in SWEEP:
    density = np.zeros(len(Xval))

    for c in classes:
        kde = fit_density(Xtr[labels == c], h)
        density += np.exp(kde.score_samples(Xval))

    ll = np.log(density + 1e-12).mean()

    if ll > best_ll:
        best_h = h
        best_ll = ll

    print(f"{h:>6.2f} {ll:>26.3f}")

print(
    f"\nvalidation selects h = {best_h:.2f} "
    f"(highest log-likelihood)"
)