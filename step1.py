import pandas as pd

# 1. بارگذاری داده‌ها
# 1. Load the dataset
data = pd.read_csv("energydata_complete.csv")

# 2. بررسی داده‌ها
# 2. Inspect the dataset
print("Initial information about the dataset:")
print(data.info())
print("\nSummary of numerical data:")
print(data.describe())
