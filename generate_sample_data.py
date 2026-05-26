"""
Generates reference.csv and current.csv with intentional drift in some columns.
Run this first to demo the detector.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

reference = pd.DataFrame({
    "age":      np.random.normal(35, 10, n).clip(18, 80).astype(int),
    "income":   np.random.normal(60000, 15000, n).clip(20000, 150000),
    "score":    np.random.beta(2, 5, n) * 100,
    "category": np.random.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
    "region":   np.random.choice(["North", "South", "East", "West"], n, p=[0.25]*4),
})

current = pd.DataFrame({
    "age":      np.random.normal(50, 12, n).clip(18, 80).astype(int),   # drifted
    "income":   np.random.normal(61000, 15500, n).clip(20000, 150000),  # stable
    "score":    np.random.beta(5, 2, n) * 100,                          # drifted
    "category": np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.2, 0.4, 0.3]),  # drifted
    "region":   np.random.choice(["North", "South", "East", "West"], n, p=[0.25]*4), # stable
})

reference.to_csv("sample_data/reference.csv", index=False)
current.to_csv("sample_data/current.csv", index=False)
print("Done — sample_data/reference.csv and sample_data/current.csv created")
