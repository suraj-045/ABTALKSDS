import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "creditcard_small.csv")
data = pd.read_csv(csv_path)

# Use first 10 customers only
customers = data[['Amount', 'Time']].head(10)

# Similarity Matrix
similarity = cosine_similarity(customers)

print("Customer Similarity Matrix:")
print(similarity)

# Choose first customer
customer_id = 0

# Similar customers
scores = list(enumerate(similarity[customer_id]))

scores = sorted(
    scores,
    key=lambda x: x[1],
    reverse=True
)

print("\nRecommended Similar Customers:")

for i in scores[1:4]:
    print("Customer", i[0], "Similarity:", round(i[1], 3))