import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Load dataset
# =========================
df = pd.read_csv("Dataset .csv")

# =========================
# FEATURE ENGINEERING (REQUIRED)
# =========================
df['Restaurant Name Length'] = df['Restaurant Name'].astype(str).str.len()
df['Address Length'] = df['Address'].astype(str).str.len()

df['Has Table Booking Encoded'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
df['Has Online Delivery Encoded'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

# =========================
# FEATURE SELECTION
# =========================
features = [
    'Price range',
    'Votes',
    'Restaurant Name Length',
    'Address Length',
    'Has Table Booking Encoded',
    'Has Online Delivery Encoded'
]

X = df[features]
y = df['Aggregate rating']

# =========================
# HANDLE MISSING VALUES
# =========================
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2  :", r2_score(y_test, y_pred))

# =========================
# CONCLUSION
# =========================

