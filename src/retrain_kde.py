import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity


SEED = 20250903

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

DBSCAN_EPS = 0.02
DBSCAN_MIN_SAMPLES = 8

BANDWIDTHS = [
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12
]


# Load the real sensor data
data = pd.read_csv("../assignment1/data/data_complete.csv")

X = data[["uv_sensor", "IR_sensor"]].to_numpy()
labels = data["label"].to_numpy()


# Split 60% train, 20% validation, 20% test.
# Stratification keeps the four lighting classes balanced.
X_train, X_rest, y_train, y_rest = train_test_split(
    X,
    labels,
    train_size=TRAIN_FRAC,
    random_state=SEED,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_rest,
    y_rest,
    test_size=0.5,
    random_state=SEED,
    stratify=y_rest
)


print("Split sizes")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))
print()


# Apply the same log transform used for DBSCAN.
X_train_log = np.log1p(X_train)
X_val_log = np.log1p(X_val)
X_test_log = np.log1p(X_test)


# Calculate scaling bounds from TRAINING DATA ONLY.
train_min = X_train_log.min(axis=0)
train_max = X_train_log.max(axis=0)


def scale_with_train_bounds(points):
    return (points - train_min) / (train_max - train_min)


X_train_scaled = scale_with_train_bounds(X_train_log)
X_val_scaled = scale_with_train_bounds(X_val_log)
X_test_scaled = scale_with_train_bounds(X_test_log)


# Discover the classes using DBSCAN on the training set only.
dbscan = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES
)

cluster_labels = dbscan.fit_predict(X_train_scaled)

classes = sorted(set(cluster_labels) - {-1})

print("Training DBSCAN")
print("Clusters:", len(classes))
print("Noise:", np.sum(cluster_labels == -1))
print()


# Show how the discovered training clusters compare with known labels.
comparison = pd.crosstab(
    y_train,
    cluster_labels,
    rownames=["True label"],
    colnames=["DBSCAN cluster"]
)

print("Training label vs DBSCAN cluster:")
print(comparison)
print()


# Select KDE bandwidth using validation data.
print("KDE bandwidth validation")
print("h\tmean log-likelihood")

best_bandwidth = None
best_score = -np.inf

for bandwidth in BANDWIDTHS:

    total_density = np.zeros(len(X_val_scaled))

    for cluster in classes:

        cluster_points = X_train_scaled[cluster_labels == cluster]

        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=bandwidth
        )

        kde.fit(cluster_points)

        total_density += np.exp(
            kde.score_samples(X_val_scaled)
        )

    mean_log_likelihood = np.log(
        total_density + 1e-12
    ).mean()

    print(
        f"{bandwidth:.2f}\t"
        f"{mean_log_likelihood:.3f}"
    )

    if mean_log_likelihood > best_score:
        best_score = mean_log_likelihood
        best_bandwidth = bandwidth


print()
print(
    "Selected bandwidth:",
    best_bandwidth
)

print(
    "Validation mean log-likelihood:",
    round(best_score, 3)
)


# Save the training scaling bounds for later model generation.
np.savez(
    "../assignment1/scaling_bounds.npz",
    minimum=train_min,
    maximum=train_max
)

print()
print("Saved scaling bounds.")