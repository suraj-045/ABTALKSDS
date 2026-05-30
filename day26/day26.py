import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("creditcard_small.csv")

# Features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Default model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

before = model.score(X_test, y_test)

print("Accuracy Before Tuning:", before)

# Hyperparameters
params = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 5, 10]
}

# Grid Search
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=3
)

grid.fit(X_train, y_train)

after = grid.best_estimator_.score(X_test, y_test)

print("Accuracy After Tuning:", after)

print("Best Parameters:")
print(grid.best_params_)
