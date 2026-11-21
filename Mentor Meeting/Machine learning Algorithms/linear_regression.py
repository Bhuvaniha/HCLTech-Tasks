# linear_regression.py
# Predicts Final_Score (a number) using Linear Regression.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def linear_regression():
    print("=" * 50)
    print("LINEAR REGRESSION")
    print("=" * 50)

    # 1. Read the common dataset
    sample = pd.read_excel("sample.xlsx")

    # 2. Features = input columns the model learns from
    X = sample[["Study_Hours", "Attendance", "Previous_Score", "Assignment_Score"]]

    # 3. Target = the number we want to predict
    y = sample["Final_Score"]

    # 4. Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Create the model
    model = LinearRegression()

    # 6. Train the model (model learns the best-fit line)
    model.fit(X_train, y_train)

    # 7. Predict Final_Score for the test students
    predictions = model.predict(X_test)

    # 8. Show actual vs predicted scores
    print("\nActual scores   :", list(y_test))
    print("Predicted scores:", [round(float(p), 1) for p in predictions])

    # 9. Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nMAE      :", round(mae, 2))
    print("R2 Score :", round(r2, 3))
    print("\nMAE = average error between real score and predicted score.")
    print("R2 Score = how well the line fits (1.0 = perfect).")

    # 10. Accuracy: how many predictions landed within 5 marks of the real score
    close = abs(y_test.values - predictions) <= 5
    accuracy = close.mean() * 100
    print("\nAccuracy:", round(accuracy, 1), "%  (predictions within 5 marks of the real score)")

    # 11. Plot: actual vs predicted scores
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions)
    plt.plot([low, high], [low, high], "r--", label="perfect prediction")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.savefig("output/linear_regression.png", bbox_inches="tight")
    print("\nPlot saved as output/linear_regression.png")

    # 12. Show the plot in a window (needs a display + tkinter)
    plt.show()


if __name__ == "__main__":
    # Run this file directly, or open main.py to choose an algorithm.
    linear_regression()
