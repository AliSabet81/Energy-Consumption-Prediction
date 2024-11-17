import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. بارگذاری داده‌های پیش‌پردازش‌شده
# 1. Load the preprocessed data
data = pd.read_csv("preprocessed_energy_data.csv")

# 2. تفکیک ویژگی‌ها (X) و خروجی (y)
# 2. Split features (X) and target (y)
X = data.drop(columns=["Appliances"])  # 'Appliances' is the target column
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

# 5. تعریف پارامترها برای بهینه‌سازی
# 5. Define the parameter grid for optimization
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (150, 100, 50)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000],
}

# 6. ایجاد مدل و استفاده از GridSearchCV
# 6. Create the model and use GridSearchCV
mlp = MLPRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=mlp, param_grid=param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=2
)

grid_search.fit(X_train, y_train)

# 7. نمایش بهترین پارامترها
# 7. Display the best parameters
print("Best parameters:")
print(grid_search.best_params_)

# 8. ارزیابی مدل با بهترین تنظیمات
# 8. Evaluate the model with the best parameters
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPerformance of the optimized model:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
