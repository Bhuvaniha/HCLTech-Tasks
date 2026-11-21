import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def logistic_regression():
    print("=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)

    sample = pd.read_excel("sample.xlsx")

    X = sample[["Study_Hours", "Attendance", "Previous_Score", "Assignment_Score"]]

    y = sample["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    print("\nPredicted classes:", [int(p) for p in predictions])
    print("Actual classes   :", list(y_test))
    print("\nProbability of FAIL and PASS for each test student:")
    for i in range(len(predictions)):
        print("  Student", i + 1, "-> pass probability:", round(probabilities[i][1], 2))

    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", round(accuracy * 100, 1), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nConfusion Matrix rows = actual, columns = predicted")



if __name__ == "__main__":
    logistic_regression()
