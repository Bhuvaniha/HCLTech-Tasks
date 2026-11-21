# naive_bayes.py
# Predicts Pass (0 or 1) using Naive Bayes (GaussianNB).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


def naive_bayes():
    print("=" * 50)
    print("NAIVE BAYES (GaussianNB)")
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

    # 5. Create the model
    #    GaussianNB assumes each feature follows a bell-curve (Gaussian) distribution
    model = GaussianNB()

    # 6. Train the model
    model.fit(X_train, y_train)

    # 7. Predict class
    predictions = model.predict(X_test)

    # 8. Predict probability
    probabilities = model.predict_proba(X_test)

    # 9. Show results
    print("\nPredicted classes:", [int(p) for p in predictions])
    print("Actual classes   :", list(y_test))
    print("\nProbability of FAIL and PASS for each test student:")
    for i in range(len(predictions)):
        print("  Student", i + 1, "-> pass probability:", round(probabilities[i][1], 2))

    # 10. Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", round(accuracy * 100, 1), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nBayes theorem: P(Pass | features) = P(features | Pass) * P(Pass)")

    # 11. Plot the two bell curves (why the model is called "Gaussian")
    passed = sample[sample["Pass"] == 1]["Study_Hours"]
    failed = sample[sample["Pass"] == 0]["Study_Hours"]

    x_line = np.linspace(0, 12, 200)

    def gaussian(data, x):
        mean = np.mean(data)
        std = np.std(data)
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


if __name__ == "__main__":
    # Run this file directly, or open main.py to choose an algorithm.
    naive_bayes()
