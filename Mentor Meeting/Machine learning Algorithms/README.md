# ML Beginner Project

A simple, beginner-level Machine Learning project with **7 algorithms** that all use **one common dataset** (`sample.xlsx`).

The goal is NOT to build a professional ML application.
The goal is to create simple working implementations that seven beginner students can understand and confidently explain to their mentor.

---

## Project Structure

```text
ML_Beginner_Project/
│
├── sample.xlsx               <- common dataset (used by all algorithms)
├── requirements.txt          <- list of required libraries
├── venv/                     <- virtual environment (created on your machine)
│
├── main.py                   <- choose which algorithm runs (comment/uncomment)
│
├── output/                   <- all generated diagrams (PNG images) go here
│
├── linear_regression.py
├── logistic_regression.py
├── decision_tree.py
├── knn.py
├── naive_bayes.py
├── svm.py
├── kmeans.py
│
├── ML_Beginner_Project_Guide.pdf   <- prep guide (code + concept + diagrams)
│
└── README.md
```

### Diagrams

Only the algorithms where the diagram is easy to explain AND actually needed produce
an image. The image is always saved inside the `output/` folder.
If your machine has a display (and `python3-tk` installed), the image also pops up in a window.

| Algorithm | Image file | What the diagram shows |
|---|---|---|
| Linear Regression | `output/linear_regression.png` | Actual vs Predicted scores |
| Logistic Regression | `output/logistic_regression_sigmoid.png` | Sigmoid curve + 0.5 threshold |
| Decision Tree | `output/decision_tree.png` | The tree itself |
| Naive Bayes | `output/naive_bayes_bell_curves.png` | Bell curves for Pass and Fail |
| K-Means | `output/kmeans_clusters.png` | Clusters + centroids |

**No diagram (text output only):**

- **KNN** — the existing plot only showed colored dots; it did not show the K neighbors
  or the vote, so it was hard to explain. The predicted vs actual classes and accuracy
  are printed instead.
- **SVM** — the existing plot showed dots and support vectors but no boundary line or
  margin, so it did not really explain SVM. Support vector count, accuracy and the
  confusion matrix are printed instead.

### Modular Design

The project is written in a modular way:

- Each algorithm lives in **its own file** and defines **one function** (example: `knn.py` defines `knn()`).
- Every function reads the same `sample.xlsx` inside the function.
- `main.py` **imports all 7 functions** and runs only the one you choose.
- Nothing is duplicated — one dataset, one function per algorithm, one place to switch.

---

## How to Run

### Step 1: Create the virtual environment (venv)

Run this once inside the project folder.

Windows:

```bash
python -m venv venv
```

Mac / Linux:

```bash
python3 -m venv venv
```

### Step 2: Activate the venv

Windows (Command Prompt / PowerShell):

```bash
venv\Scripts\activate
```

Mac / Linux:

```bash
source venv/bin/activate
```

You will see `(venv)` at the start of your prompt. Keep it activated while working.

### Step 3: Install the libraries from requirements.txt

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `openpyxl`.

**Note:** the `venv` folder only exists on your machine. Anyone else can recreate it
with Steps 1–3 and the same `requirements.txt`.

### Step 4: Run from main.py (recommended)

Open `main.py`, comment / uncomment the algorithm you want, then:

```bash
python main.py
```

### Step 5: Or run any file directly

```bash
python linear_regression.py
python logistic_regression.py
python decision_tree.py
python knn.py
python naive_bayes.py
python svm.py
python kmeans.py
```

---

## Preparation Guide (PDF)

`ML_Beginner_Project_Guide.pdf` is the mentor-prep document. It explains each program in three parts:
- **Concept** — what the algorithm does and its key terms (in simple words)
- **Code** — the code broken down step by step
- **Diagram** — what the PNG image shows and how to read it (for the algorithms that have one)

It also includes the comparison table, Regression vs Classification, Supervised vs Unsupervised,
and the team division with key points per member.

---

## Comment / Uncomment Pattern (in main.py)

`main.py` imports all 7 algorithms, then has this block at the bottom:

```python
if __name__ == "__main__":
    linear_regression()
    # logistic_regression()
    # decision_tree()
    # knn()
    # naive_bayes()
    # svm()
    # kmeans()
```

**Only the line WITHOUT `#` runs.**

In the example above, only Linear Regression runs. If you change it to:

```python
if __name__ == "__main__":
    # linear_regression()
    # logistic_regression()
    # decision_tree()
    # knn()
    # naive_bayes()
    # svm()
    kmeans()
```

then only K-Means runs.

**Note:** every algorithm file can also be run directly (example: `python svm.py`),
because each file has its own `if __name__ == "__main__":` block at the bottom.

---

## The Common Dataset

One dataset (`sample.xlsx`) is used by ALL 7 algorithms.
No algorithm creates its own dataset — every file reads the same Excel file:

```python
sample = pd.read_excel("sample.xlsx")
```

| Column | Meaning |
|---|---|
| Study_Hours | Hours of study per day |
| Attendance | Attendance percentage |
| Previous_Score | Score in the previous exam |
| Assignment_Score | Average assignment score |
| Sleep_Hours | Hours of sleep per night |
| Final_Score | Final exam score (0-100) |
| Pass | 0 = Fail, 1 = Pass (Final_Score 50 or more) |

100 student records. Features are realistic and consistent: more study + better attendance + higher previous score → higher final score.

---

## 1. Linear Regression

**File:** `linear_regression.py`
**Target:** `Final_Score` (a number)
**Type:** Regression, Supervised Learning

### Concepts

- **What is regression?** Predicting a number.
- **What is Linear Regression?** It finds a straight line (best-fit line) that fits the data.
- **What are features?** The input columns: `Study_Hours`, `Attendance`, `Previous_Score`, `Assignment_Score`.
- **What is the target?** The output we want to predict: `Final_Score`.
- **Why is Final_Score the target?** Because it is a continuous number, and the other columns are what we know before the exam.
- **What does the model learn?** It learns how much each feature changes the Final_Score (the slope of the line).
- **What is a best-fit line?** The line that stays as close as possible to all the data points.
- **What is prediction?** Giving the trained model new input values and getting a number back.

### Evaluation

- **MAE** = average error between real score and predicted score.
- **R² Score** = how well the line fits (1.0 = perfect).
- **Accuracy** = how many test predictions landed within 5 marks of the real score.

### How I explain this to my mentor

"I am using Linear Regression because my target is a continuous numerical value, which is `Final_Score`. I use study hours, attendance, previous score, and assignment score as features. The model learns the relationship between these features and the final score using the training data. Then I use the trained model to predict the score for new data. I evaluate it with MAE, which tells me the average prediction error, and R² Score, which tells me how well the line fits."

---

## 2. Logistic Regression

**File:** `logistic_regression.py`
**Target:** `Pass` (0 or 1)
**Type:** Classification, Supervised Learning

**Important:** the correct name is **Logistic Regression**, not "Logistics Regression".

### Concepts

- **Classification** = predicting a category (Pass/Fail), not a number.
- **Probability** = how sure the model is (0.0 to 1.0).
- **Sigmoid function** = a curve that converts any number into a value between 0 and 1.
- **Threshold** = if probability ≥ 0.5 → class 1 (Pass), else class 0 (Fail).
- **Why the output is 0 or 1?** Because the sigmoid output is cut at the threshold to give a class.
- **Difference from Linear Regression:** Linear Regression predicts a number, Logistic Regression predicts a category.

### Evaluation

- **Accuracy** = how many predictions were correct.
- **Confusion Matrix** = table showing correct and wrong predictions for each class.

### How I explain this to my mentor

"I am using Logistic Regression because my target is a category: `Pass`, which is 0 or 1. Even though it has 'Regression' in the name, it is a classification algorithm. It uses the sigmoid function to convert the input into a probability between 0 and 1, and then a threshold of 0.5 decides if the student passes or fails. I show both `predict()` for the class and `predict_proba()` for the probability. I evaluate it with Accuracy and a Confusion Matrix."

---

## 3. Decision Tree

**File:** `decision_tree.py`
**Target:** `Pass`
**Type:** Classification, Supervised Learning

### Concepts

- **Root node** = the first question at the top of the tree (example: "Attendance ≤ 67.5?").
- **Decision node** = a node that asks a question.
- **Branch** = the arrow going left or right (Yes or No).
- **Leaf node** = the final node that gives the answer (Pass or Fail).
- **How the tree makes decisions** = it starts at the root, asks questions, and follows branches until it reaches a leaf.
- **Gini impurity** = a number from 0 to 1. 0 = pure (all students same class), 0.5 = mixed (half pass, half fail). The tree keeps splitting to make each node purer.

We use `max_depth=3` so the tree stays small and easy to read.
We draw it with `plot_tree()`.

### How I explain this to my mentor

"I am using a Decision Tree because it works like a flowchart of questions, which is very easy to explain. The tree picks the best feature to split on first, for example attendance, then splits again on the next best feature. Each split aims to reduce Gini impurity, meaning each group becomes purer. I set `max_depth=3` so the tree is small enough to read, and I visualize it with `plot_tree()`. The final leaf node gives the prediction: Pass or Fail."

---

## 4. KNN (K-Nearest Neighbors)

**File:** `knn.py`
**Target:** `Pass`
**Type:** Classification, Supervised Learning

### Concepts

- **What KNN means** = K-Nearest Neighbors: look at the K closest students in the data.
- **What is a neighbor?** A student from the training data whose feature values are closest to the student we are predicting.
- **What is K?** How many neighbors we look at. We use `K=3`.
- **Distance** = how far two students are in feature space (we use Euclidean distance).
- **Voting** = the majority of the K neighbors wins (2 out of 3 say Pass → prediction is Pass).
- **Why K=3?** Small, odd number (odd avoids ties), easy to explain.
- **K too small** = the model memorizes noise and overfits.
- **K too large** = the model becomes too general and misses patterns.
- **StandardScaler** = puts every column on the same range (mean 0, standard deviation 1). KNN uses distance, so without scaling, `Attendance` (0-100) would matter much more than `Study_Hours` (0-10).

### How I explain this to my mentor

"I am using KNN because it is the simplest classification idea: you are similar to your neighbors. For a new student, KNN finds the 3 closest students from the training data and takes a majority vote of their Pass/Fail labels. Before that, I use `StandardScaler` so all features are on the same scale, because KNN measures distance. If K is too small the model overfits, and if K is too large it becomes too general. I evaluate it with Accuracy."

---

## 5. Naive Bayes

**File:** `naive_bayes.py`
**Target:** `Pass`
**Type:** Classification, Supervised Learning

### Concepts

- **Probability** = how likely something is (0 to 1).
- **Bayes theorem** = a formula to update a probability with new evidence:

  ```text
  P(Pass | features) = P(features | Pass) x P(Pass) / P(features)
  ```

  In simple words: "Given these study habits, how likely is this student to pass?"
- **Why it is called "Naive"?** Because it assumes every feature is independent of the others (study hours do not affect attendance). That assumption is usually wrong in real life — that is why it is "naive".
- **Independence assumption** = features do not influence each other.
- **GaussianNB** = the version of Naive Bayes that assumes each feature follows a bell-curve (Gaussian) distribution.

### How I explain this to my mentor

"I am using Naive Bayes because it is a fast, probability-based classifier built on Bayes theorem. It calculates the probability of Pass and Fail for a student and picks the higher one. It is called 'naive' because it assumes all features are independent, which is not really true, but it still works well in practice. I use `GaussianNB` because my features are continuous numbers, and Gaussian means the bell-curve distribution. I show both `predict()` and `predict_proba()` so the mentor can see the probabilities behind the prediction."

---

## 6. SVM (Support Vector Machine)

**File:** `svm.py`
**Target:** `Pass`
**Type:** Classification, Supervised Learning

### Concepts

- **What SVM means** = Support Vector Machine, a classification algorithm.
- **Classification** = separating two classes (Pass and Fail).
- **Hyperplane** = the line (or boundary) that separates the two classes.
- **Margin** = the empty space between the hyperplane and the nearest students of each class. SVM tries to make this margin as wide as possible.
- **Support vectors** = the students closest to the hyperplane. They are the only points that actually decide where the line goes.
- **kernel="linear"** = a straight boundary line (kept simple, no advanced math).
- **StandardScaler** = SVM works with distances too, so scaling the features puts all columns on the same range and helps the hyperplane be found correctly.

### How I explain this to my mentor

"I am using SVM because it finds the best boundary line between Pass and Fail students. The algorithm looks for the hyperplane with the widest possible margin between the two classes. The students closest to that line are called support vectors, and they define the boundary. I use a linear kernel to keep it simple and `StandardScaler` so all features are on the same scale. I evaluate it with Accuracy and a Confusion Matrix."

---

## 7. K-Means Clustering

**File:** `kmeans.py`
**Target:** none — this is **unsupervised learning**
**Type:** Clustering, Unsupervised Learning

**Important:** K-Means does NOT use the `Pass` column. There is no target.

Features used: `Study_Hours`, `Attendance`, `Previous_Score`.

### Concepts

- **What is clustering?** Grouping similar things together without any labels.
- **What is unsupervised learning?** The model finds patterns by itself — there is no answer column to learn from.
- **What does K mean?** The number of groups. We use `K=3` → 3 clusters.
- **What is a centroid?** The center point of a cluster (the average of all students in it).
- **How are students assigned?** Each student goes to the cluster whose centroid is closest.
- **How are centroids updated?** After all students are assigned, each centroid moves to the average position of its students.
- **How does the process repeat?** Assign students → update centroids → assign again → repeat until the centroids stop moving.
- **inertia_** = the total distance of all students from their own cluster center. Smaller inertia means students are packed more tightly into their clusters.
- **Silhouette Score** = how well separated the clusters are, from -1 to 1. Close to 1 = clear, well-separated clusters.
- **Accuracy (check only)** = after clustering, each cluster is compared with the real `Pass` column. The accuracy is the percentage of students whose cluster matches their real Pass/Fail. `Pass` is not used to train the model — only to check the result.

### How I explain this to my mentor

"I am using K-Means because this part of the project is about unsupervised learning, so there is no target column like Pass or Final_Score. The model groups the 100 students into 3 clusters based on study hours, attendance, and previous score. It starts with 3 random centroids, assigns every student to the closest centroid, then updates the centroids and repeats until they stop moving. I show the cluster assignments, the centroids, and the inertia, which tells me how tightly the students are grouped. I also plot the clusters on a scatter plot so the mentor can see the three groups visually."

---

## Final Comparison Table

| Algorithm | Type | Learning | Target/Goal |
|---|---|---|---|
| Linear Regression | Regression | Supervised | Final_Score |
| Logistic Regression | Classification | Supervised | Pass |
| Decision Tree | Classification | Supervised | Pass |
| KNN | Classification | Supervised | Pass |
| Naive Bayes | Classification | Supervised | Pass |
| SVM | Classification | Supervised | Pass |
| K-Means | Clustering | Unsupervised | Student Groups |

### Regression vs Classification

- **Regression** = the target is a number (example: Final_Score = 74.3).
- **Classification** = the target is a category (example: Pass = 1 or Fail = 0).
- In this project, only Linear Regression is regression. The other five supervised algorithms are classification.

### Supervised vs Unsupervised Learning

- **Supervised learning** = the data has an answer column (a target). The model learns from inputs → target pairs. Example: study habits → Pass.
- **Unsupervised learning** = there is no answer column. The model finds hidden patterns by itself, like grouping similar students. Example: K-Means creating 3 student groups.

### Why the same `sample.xlsx` dataset can be used for different algorithms

- The dataset is just a table of student data. An algorithm only fails to use a column if we choose not to pass it in.
- For regression we set `y = Final_Score` (a number).
- For classification we set `y = Pass` (a category).
- For clustering we use no `y` at all, only the feature columns.
- So the same rows of data can be used for all three tasks — we just pick different columns as the target (or no target at all).

---

## Team Division

Each team member explains ONLY one algorithm.

### Member 1 → Linear Regression

1. Target is `Final_Score`, a continuous number → this is regression, not classification.
2. Features: Study_Hours, Attendance, Previous_Score, Assignment_Score.
3. The model learns a best-fit line that maps features → score.
4. Evaluation: MAE (average error) and R² Score (how well the line fits, max 1.0).
5. Uses `LinearRegression()` with `fit()` then `predict()`.

### Member 2 → Logistic Regression

1. Target is `Pass` (0 or 1) → classification, even though it has "Regression" in the name.
2. Uses the sigmoid function to turn output into a probability between 0 and 1.
3. Threshold: probability ≥ 0.5 → Pass (1), otherwise Fail (0).
4. Shows `predict()` and `predict_proba()`.
5. Evaluation: Accuracy and Confusion Matrix.

### Member 3 → Decision Tree

1. Target is `Pass`; the tree works like a flowchart of questions.
2. Know the parts: root node, decision node, branch, leaf node.
3. Gini impurity measures how mixed a node is (0 = pure, 0.5 = half-half). The tree splits to make nodes purer.
4. `max_depth=3` keeps the tree small and easy to explain.
5. Visualized with `plot_tree()`.

### Member 4 → KNN

1. Target is `Pass`; prediction = majority vote of the K closest students.
2. `n_neighbors=3`: K is small and odd (odd avoids ties).
3. Distance is used, so `StandardScaler` is applied first.
4. K too small → overfits noise. K too large → too general.
5. Evaluation: Accuracy.

### Member 5 → Naive Bayes

1. Target is `Pass`; based on Bayes theorem (probability with evidence).
2. Called "naive" because it assumes all features are independent (they are not, but it still works).
3. `GaussianNB` = assumes each feature follows a bell-curve distribution.
4. Shows `predict()` and `predict_proba()`.
5. Evaluation: Accuracy and Confusion Matrix.

### Member 6 → SVM

1. Target is `Pass`; SVM finds the best separating line (hyperplane).
2. Margin = the gap between classes; SVM makes it as wide as possible.
3. Support vectors = the students closest to the hyperplane; they define it.
4. `kernel="linear"` = straight boundary; `StandardScaler` puts features on the same scale.
5. Evaluation: Accuracy and Confusion Matrix.

### Member 7 → K-Means

1. Unsupervised — NO target column, `Pass` is NOT used.
2. K = number of clusters (3); a centroid is the center of a cluster.
3. Process: assign students → update centroids → repeat until centroids stop moving.
4. `inertia_` = total distance of students from their centroids (smaller = tighter clusters); also Silhouette Score (closer to 1 = better).
5. Result shown via `sample["Cluster"] = model.labels_` and a scatter plot.

---

## Final Check

- [x] `sample.xlsx` is present in the project folder.
- [x] `requirements.txt` lists all required libraries.
- [x] `venv` is created and all requirements are installed.
- [x] All algorithms use the SAME `sample.xlsx`.
- [x] No algorithm creates its own dataset.
- [x] Each algorithm is modular (one function per file, imported by `main.py`).
- [x] Each algorithm can be run independently (`python knn.py`).
- [x] `main.py` has the comment/uncomment block — only the uncommented algorithm executes.
- [x] The code is beginner-level.
- [x] The code is easy to explain line by line.
- [x] Each team member can focus on one algorithm.
- [x] The project is suitable for a mentor meeting.
