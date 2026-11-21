# knn.py
# Predicts Pass (0 or 1) using K-Nearest Neighbors.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn():
    print("=" * 50)
    print("K-NEAREST NEIGHBORS (KNN)")
    print("=" * 50)

    # 1. Read the common dataset
    sample = pd.read_excel("sample.xlsx")

    # 2. Features (inputs)
    X = sample[["Study_Hours", "Attendance", "Previous_Score", "Assignment_Score"]]

    # 3. Target: Pass = 0 or 1
    y = sample["Pass"]

    # 4. Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Scale the features so all columns are on the same range.
    #    KNN measures distance, so columns like Attendance (0-100)
    #    would otherwise dominate Study_Hours (0-10).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. Create the model: look at the 3 nearest students (K=3)
    model = KNeighborsClassifier(n_neighbors=3)

    # 7. Train the model
    model.fit(X_train, y_train)

    # 8. Predict
    predictions = model.predict(X_test)

    # 9. Show results
    print("\nPredicted classes:", [int(p) for p in predictions])
    print("Actual classes   :", list(y_test))

    # 10. Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", round(accuracy * 100, 1), "%")
    print("\nEach prediction = majority vote of the 3 closest students.")


if __name__ == "__main__":
    # Run this file directly, or open main.py to choose an algorithm.
    knn()
