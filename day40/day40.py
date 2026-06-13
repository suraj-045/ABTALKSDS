import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -------------------------
# Simulate A/B Test Data
# -------------------------

np.random.seed(42)

control = np.random.normal(
    loc=50,
    scale=10,
    size=100
)

experiment = np.random.normal(
    loc=55,
    scale=10,
    size=100
)

# -------------------------
# Calculate Averages
# -------------------------

control_mean = control.mean()
experiment_mean = experiment.mean()

print("Control Group Average:")
print(round(control_mean, 2))

print("\nExperiment Group Average:")
print(round(experiment_mean, 2))

# -------------------------
# Statistical Significance
# -------------------------

t_stat, p_value = ttest_ind(
    control,
    experiment
)

print("\nP-Value:")
print(round(p_value, 5))

# -------------------------
# Decision
# -------------------------

if p_value < 0.05:
    print("\nResult: Significant Difference")
else:
    print("\nResult: No Significant Difference")

# -------------------------
# Visualization
# -------------------------

plt.boxplot(
    [control, experiment],
    labels=["Control", "Experiment"]
)

plt.title("A/B Test Results")
plt.ylabel("Conversion Score")

plt.show()

print("\nA/B Testing Analysis Completed!")