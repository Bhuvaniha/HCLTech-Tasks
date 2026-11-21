# decision_tree.py
# Predicts Pass (0 or 1) using a Decision Tree.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def decision_tree():
    print("=" * 50)
    print("DECISION TREE")
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

    # 5. Create the model (max_depth=3 keeps the tree small and easy to read)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)

    # 6. Train the model
    model.fit(X_train, y_train)

    # 7. Predict
    predictions = model.predict(X_test)

    # 8. Show results
    print("\nPredicted classes:", [int(p) for p in predictions])
    print("Actual classes   :", list(y_test))

    # 9. Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", round(accuracy * 100, 1), "%")

    # 10. Draw the tree
    plt.figure(figsize=(12, 6))
    plot_tree(
        model,
        feature_names=list(X.columns),
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree - Pass or Fail")

    # 11. Save the plot as an image inside the output folder
    plt.savefig("output/decision_tree.png", bbox_inches="tight")
    print("\nTree plot saved as output/decision_tree.png")

    # 12. Show the plot in a window (needs a display + tkinter)
    plt.show()


if __name__ == "__main__":
    # Run this file directly, or open main.py to choose an algorithm.
    decision_tree()
