import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load Dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Feature Engineering
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

# Segmentation
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

data["Segment"] = kmeans.fit_predict(
    data[["Amount", "Time"]]
)

# Churn Label
data["Churn"] = (
    data["HighAmount"] == 0
).astype(int)

X = data[["Amount", "Time", "Segment"]]
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Baseline Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

baseline_pred = lr.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

# Optimized Model
rf = RandomForestClassifier(random_state=42)

params = {
    "n_estimators": [10, 50, 100],
    "max_depth": [2, 4, 6]
}

grid = GridSearchCV(
    rf,
    params,
    cv=3
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

optimized_pred = best_model.predict(X_test)

optimized_accuracy = accuracy_score(
    y_test,
    optimized_pred
)

print("Baseline Accuracy:")
print(round(baseline_accuracy,4))

print("\nOptimized Accuracy:")
print(round(optimized_accuracy,4))

print("\nBest Parameters:")
print(grid.best_params_)