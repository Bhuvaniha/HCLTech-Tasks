import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def svm():
    print("=" * 50)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 50)

    sample = pd.read_excel("sample.xlsx")

    X = sample[["Study_Hours", "Attendance", "Previous_Score", "Assignment_Score"]]

    y = sample["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="linear")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nPredicted classes:", [int(p) for p in predictions])
    print("Actual classes   :", list(y_test))

    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", round(accuracy * 100, 1), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nSupport vectors = the students closest to the boundary line.")
    print("Number of support vectors:", model.n_support_)


if __name__ == "__main__":
    svm()
