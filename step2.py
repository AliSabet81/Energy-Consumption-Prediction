import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. بارگذاری داده‌ها
# 1. Load the dataset
data = pd.read_csv("energydata_complete.csv")

# 2. بررسی داده‌ها
# 2. Inspect the dataset
print("Initial information about the dataset:")
print(data.info())
print("\nSummary of numerical data:")
print(data.describe())

# 3. بررسی داده‌های گمشده
print("\nNumber of missing values in each column:")
print(data.isnull().sum())

# اگر داده گمشده‌ای وجود داشته باشد، می‌توان آن را مدیریت کرد
# حذف داده‌های گمشده
# If there are any missing values, handle them
# Removing rows with missing values
data.dropna(inplace=True)

# 4. حذف ویژگی‌های غیرضروری
# 4. Drop unnecessary features
columns_to_drop = ["date", "rv1", "rv2"]
data = data.drop(columns=columns_to_drop)
print("\nRemaining columns:")
print(data.columns)

# 5. نرمال‌سازی داده‌ها
# 5. Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data_normalized = pd.DataFrame(scaled_data, columns=data.columns)

# نمایش داده‌های نرمال‌سازی شده
# Display normalized data
print("\nNormalized data (first few rows):")
print(data_normalized.head())

# ذخیره داده‌های پیش‌پردازش شده
# Save the preprocessed data
data_normalized.to_csv("preprocessed_energy_data.csv", index=False)
print("\nPreprocessed data saved as: 'preprocessed_energy_data.csv'")
