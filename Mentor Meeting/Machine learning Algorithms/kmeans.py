# kmeans.py
# Groups students into clusters using K-Means.
# NOTE: K-Means is UNSUPERVISED learning - there is NO target column.
# We do NOT use "Pass" here.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans():
    print("=" * 50)
    print("K-MEANS CLUSTERING")
    print("=" * 50)

    # 1. Read the common dataset
    sample = pd.read_excel("sample.xlsx")

    # 2. Features only - no target, because this is unsupervised learning
    X = sample[["Study_Hours", "Attendance", "Previous_Score"]]

    # 3. Create the model: split students into 3 groups
    model = KMeans(n_clusters=3, random_state=42)

    # 4. Train the model (K-Means finds the groups by itself)
    model.fit(X)

    # 5. Add the cluster number of each student to the dataset
    sample["Cluster"] = model.labels_

    # 6. Print the result
    print("\nStudents grouped into clusters:")
    print(sample[["Study_Hours", "Attendance", "Previous_Score", "Cluster"]])

    # 7. Show the center (centroid) of each cluster
    print("\nCluster centers (centroids):")
    print(pd.DataFrame(model.cluster_centers_,
                       columns=["Study_Hours", "Attendance", "Previous_Score"]).round(1))

    # 8. Explain inertia: total distance of all students from their cluster center.
    #    Smaller inertia = students are closer to their own cluster center.
    print("\nInertia:", round(model.inertia_, 1))
    print("(Inertia = how far students are from their cluster center. Smaller is better.)")

    # 9. Silhouette Score: how well separated the clusters are (-1 to 1, higher is better)
    sil = silhouette_score(X, model.labels_)
    print("Silhouette Score:", round(sil, 3))
    print("(Silhouette Score close to 1 = clearly separated clusters.)")

    # 10. Accuracy: check the clusters against the real Pass/Fail column.
    #     "Pass" is NOT used to train the model - only to check the result afterwards.
    #     Each cluster is given the majority Pass/Fail label of its own students,
    #     then we count how many students match that label.
    majority = sample.groupby("Cluster")["Pass"].agg(lambda s: s.mode()[0])
    matched = sample["Cluster"].map(majority) == sample["Pass"]
    accuracy = matched.mean() * 100
    print("\nAccuracy:", round(accuracy, 1), "%  (students whose cluster matches their real Pass/Fail)")

    # 11. Simple scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(sample["Study_Hours"], sample["Attendance"],
                c=sample["Cluster"], cmap="viridis", s=120)
    plt.xlabel("Study Hours")
    plt.ylabel("Attendance")
    plt.title("K-Means: Students grouped into 3 clusters")
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                marker="X", s=200, c="red", label="centroids")
    plt.legend()
    plt.colorbar(label="Cluster")

    # 12. Save the plot as an image inside the output folder
    plt.savefig("output/kmeans_clusters.png", bbox_inches="tight")
    print("\nCluster plot saved as output/kmeans_clusters.png")

    # 13. Show the plot in a window (needs a display + tkinter)
    plt.show()


if __name__ == "__main__":
    # Run this file directly, or open main.py to choose an algorithm.
    kmeans()
