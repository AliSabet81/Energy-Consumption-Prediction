import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. بارگذاری داده‌های پیش‌پردازش‌شده
# 1. Load the preprocessed data
data = pd.read_csv("preprocessed_energy_data.csv")

# 2. تفکیک ویژگی‌ها (X) و خروجی (y)
# 2. Split features (X) and target (y)
X = data.drop(columns=["Appliances"])
y = data["Appliances"]

# 3. تقسیم داده‌ها به مجموعه‌های آموزش و تست
# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. نرمال‌سازی داده‌ها
# 4. Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. ایجاد مدل شبکه عصبی بهینه‌سازی‌شده
# 5. Create the optimized neural network model
best_mlp = MLPRegressor(
    hidden_layer_sizes=(150, 100, 50),
    activation="relu",
    solver="adam",
    learning_rate="constant",
    max_iter=500,
    random_state=42,
)

# 6. آموزش مدل
# 6. Train the model
best_mlp.fit(X_train, y_train)

# 7. پیش‌بینی با مدل
# 7. Make predictions with the model
y_pred = best_mlp.predict(X_test)

# 8. محاسبه معیارهای ارزیابی
# 8. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# 9. رسم نمودار مقایسه مقادیر واقعی و پیش‌بینی‌شده
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y_test)), y_test, label="Actual Values", color="blue", alpha=0.7)
plt.plot(np.arange(len(y_pred)), y_pred, label="Predictions", color="red", alpha=0.7)
plt.legend()
plt.title("Comparison of Actual and Predicted Values")
plt.xlabel("Samples")
plt.ylabel("Energy Consumption")
plt.show()

# 10. رسم نمودار پراکندگی
# 10. Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="purple")
plt.title("Scatter Plot of Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2, color="red"
)
plt.show()
