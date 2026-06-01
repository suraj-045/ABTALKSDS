import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("creditcard_small.csv")

# Features and Target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Base Model
# -----------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

base_accuracy = model.score(X_test, y_test)

print("Base Model Accuracy:")
print(base_accuracy)

# -----------------------------
# Cross Validation
# -----------------------------
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\nCross Validation Average:")
print(scores.mean())

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
params = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=3
)

grid.fit(X_train, y_train)

# Best Model
best_model = grid.best_estimator_

optimized_accuracy = best_model.score(
    X_test,
    y_test
)

print("\nOptimized Model Accuracy:")
print(optimized_accuracy)

print("\nBest Parameters:")
print(grid.best_params_)