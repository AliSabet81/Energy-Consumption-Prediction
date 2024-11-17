import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. بارگذاری داده‌های پیش‌پردازش‌شده
# 1. Load Preprocessed Data
data = pd.read_csv("preprocessed_energy_data.csv")

# 2. تفکیک ویژگی‌ها (X) و خروجی (y)
# 2. Separate Features (X) and Target (y)
X = data.drop(columns=["Appliances"])
y = data["Appliances"]

# 3. تقسیم داده‌ها به مجموعه‌های آموزش و تست
# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. نرمال‌سازی داده‌ها
# 4. Normalize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. ایجاد مدل‌ها
# رگرسیون خطی
# 5. Create Models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# درخت تصمیم
# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# شبکه عصبی
# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)


# 6. ارزیابی مدل‌ها
# 6. Evaluate Models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")


evaluate_model(y_test, lr_predictions, "Linear Regression")
evaluate_model(y_test, dt_predictions, "Decision Tree")
evaluate_model(y_test, nn_predictions, "Neural Network")

# 7. نتیجه‌گیری
# 7. Conclusion
print(
    "\nModels have been evaluated. Choose the best model based on accuracy and lower error."
)
