import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


# Load the real labelled sensor dataset
data = pd.read_csv("../assignment1/data/data_complete.csv")

# DBSCAN uses only the two sensor readings.
# Labels are kept separately and are only used afterward for evaluation.
sensor_data = data[["uv_sensor", "IR_sensor"]]
true_labels = data["label"]


print("Dataset size:", len(data))
print()

print("Samples per class:")
print(true_labels.value_counts())
print()


# The sunlight readings are much larger than the indoor readings.
# log1p compresses this large numeric range while preserving the order
# of the sensor values. It also works safely when a reading is zero.
transformed_data = np.log1p(sensor_data)


# Scale both transformed features to 0-1 before DBSCAN.
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(transformed_data)


# Final DBSCAN settings selected from the parameter sweep.
dbscan = DBSCAN(
    eps=0.02,
    min_samples=8
)

cluster_labels = dbscan.fit_predict(scaled_data)


# Count discovered clusters and noise points.
unique_clusters = set(cluster_labels)
number_of_clusters = len(unique_clusters - {-1})
number_of_noise_points = sum(cluster_labels == -1)

print("Final DBSCAN result")
print("eps = 0.02")
print("min_samples = 8")
print("Clusters discovered:", number_of_clusters)
print("Noise points:", number_of_noise_points)


# Required comparison between the known lighting labels
# and the clusters discovered without using those labels.
confusion_table = pd.crosstab(
    true_labels,
    cluster_labels,
    rownames=["True label"],
    colnames=["DBSCAN cluster"]
)

print()
print("True label vs DBSCAN cluster:")
print(confusion_table)


# Save the discovered cluster number with each original sample.
# This will be used by the KDE stage.
clustered_data = data.copy()
clustered_data["cluster"] = cluster_labels

clustered_data.to_csv(
    "../assignment1/data/data_clustered.csv",
    index=False
)

print()
print("Saved clustered data to:")
print("../assignment1/data/data_clustered.csv")