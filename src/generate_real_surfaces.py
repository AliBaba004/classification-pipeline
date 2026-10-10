import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity


SEED = 20250903

GRID_N = 16
BANDWIDTH = 0.03

DBSCAN_EPS = 0.02
DBSCAN_MIN_SAMPLES = 8


# Load the real sensor dataset
data = pd.read_csv("../assignment1/data/data_complete.csv")

X = data[["uv_sensor", "IR_sensor"]].to_numpy()
true_labels = data["label"].to_numpy()


# Use the same 60/20/20 split as the KDE validation step.
X_train, X_rest, y_train, y_rest = train_test_split(
    X,
    true_labels,
    train_size=0.60,
    random_state=SEED,
    stratify=true_labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_rest,
    y_rest,
    test_size=0.5,
    random_state=SEED,
    stratify=y_rest
)


# Apply the same log transform used during DBSCAN tuning.
X_train_log = np.log1p(X_train)
X_val_log = np.log1p(X_val)
X_test_log = np.log1p(X_test)


# Scaling bounds must come from the training set only.
train_min = X_train_log.min(axis=0)
train_max = X_train_log.max(axis=0)


def scale_with_train_bounds(points):
    return (points - train_min) / (train_max - train_min)


X_train_scaled = scale_with_train_bounds(X_train_log)
X_val_scaled = scale_with_train_bounds(X_val_log)
X_test_scaled = scale_with_train_bounds(X_test_log)


# Run DBSCAN on the training data only.
dbscan = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES
)

cluster_labels = dbscan.fit_predict(X_train_scaled)

clusters = sorted(set(cluster_labels) - {-1})

print("Clusters discovered:", len(clusters))
print("Noise points:", np.sum(cluster_labels == -1))
print()


# Determine which real lighting condition corresponds to each
# discovered cluster. The labels are only used after clustering.
class_names = []

for cluster in clusters:

    labels_in_cluster = y_train[cluster_labels == cluster]

    most_common_label = pd.Series(
        labels_in_cluster
    ).value_counts().idxmax()

    class_names.append(most_common_label)

    print(
        "Cluster",
        cluster,
        "->",
        most_common_label,
        "(",
        len(labels_in_cluster),
        "training samples )"
    )


# Create the 16 x 16 lookup grid.
grid_values = np.linspace(
    0.0,
    1.0,
    GRID_N
)

grid_x, grid_y = np.meshgrid(
    grid_values,
    grid_values
)

grid_points = np.column_stack(
    (
        grid_x.ravel(),
        grid_y.ravel()
    )
)


continuous_surfaces = []
byte_surfaces = []

print()
print("Generating KDE surfaces...")


for cluster in clusters:

    cluster_points = X_train_scaled[
        cluster_labels == cluster
    ]

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=BANDWIDTH
    )

    kde.fit(cluster_points)

    log_density = kde.score_samples(grid_points)

    density = np.exp(log_density)

    surface = density.reshape(
        GRID_N,
        GRID_N
    )


    # Normalize each class surface to its own peak,
    # matching the previous lookup-table pipeline.
    peak = surface.max()

    if peak > 0:
        normalized_surface = surface / peak
    else:
        normalized_surface = surface


    # Convert the normalized values to bytes.
    byte_surface = np.round(
        normalized_surface * 255
    ).astype(np.uint8)


    continuous_surfaces.append(
        normalized_surface
    )

    byte_surfaces.append(
        byte_surface
    )


continuous_surfaces = np.array(
    continuous_surfaces
)

byte_surfaces = np.array(
    byte_surfaces
)


# Measure the quantization error.
restored_surfaces = byte_surfaces.astype(
    np.float64
) / 255.0

quantization_error = np.max(
    np.abs(
        continuous_surfaces
        - restored_surfaces
    )
)


# Save everything needed for the later model.h generator.
np.savez(
    "../assignment1/surfaces.npz",
    continuous=continuous_surfaces,
    byte_tables=byte_surfaces,
    class_names=np.array(class_names),
    grid_n=GRID_N,
    bandwidth=BANDWIDTH
)


# Save scaling bounds separately because Part 3
# needs them when generating model.h.
np.savez(
    "../assignment1/scaling_bounds.npz",
    minimum=train_min,
    maximum=train_max
)


# Report memory footprint.
num_classes = len(clusters)

bytes_per_class = GRID_N * GRID_N
total_table_bytes = (
    num_classes
    * bytes_per_class
)

bounds_bytes = 4 * 4


print()
print("Stored model")
print("Grid:", GRID_N, "x", GRID_N)
print("Classes:", num_classes)
print("Bytes per class:", bytes_per_class)
print("All class tables:", total_table_bytes, "bytes")
print("Scaling bounds:", bounds_bytes, "bytes")
print(
    "Total tables + bounds:",
    total_table_bytes + bounds_bytes,
    "bytes"
)

print()
print(
    "Worst-case uint8 quantization error:",
    f"{quantization_error * 100:.3f}%"
)

print()
print("Saved:")
print("../assignment1/surfaces.npz")
print("../assignment1/scaling_bounds.npz")