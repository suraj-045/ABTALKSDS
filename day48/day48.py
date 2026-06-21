import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load Dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

print("Original Shape:")
print(data.shape)

# Missing Value Handling
imputer = SimpleImputer(strategy="mean")

data[["Amount"]] = imputer.fit_transform(
    data[["Amount"]]
)

# Feature Engineering
data["HighAmount"] = (
    data["Amount"] > data["Amount"].mean()
).astype(int)

# Scaling
scaler = StandardScaler()

data[["Amount"]] = scaler.fit_transform(
    data[["Amount"]]
)

# Save Clean Dat
data.to_csv(
    "cleaned_creditcard.csv",
    index=False
)

print("\nCleaned Shape:")
print(data.shape)

print("\nPipeline Completed!")