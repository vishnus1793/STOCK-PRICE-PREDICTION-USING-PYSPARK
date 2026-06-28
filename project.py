from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 1. Download and Prepare Historical Data
# ==========================================
print("Fetching market data...")
raw_data = yf.download(
    "GOOGL", start="2010-01-01", end=date.today().strftime("%Y-%m-%d")
)

# CRITICAL FIX: Flatten any multi-level columns returned by yfinance
data = raw_data.copy()
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Feature Engineering
data["Date"] = data.index
data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year
data["Previous_close"] = data["Close"].shift(1)
data.dropna(inplace=True)

# Select features (X) and target variable (y)
X = data[["Previous_close", "Day", "Month", "Year"]]
y = data["Close"]

# ==========================================
# 2. Time-Series Cross-Validation & Training
# ==========================================
print("\nTraining Model via TimeSeriesSplit...")
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = lr()
    model.fit(X_train.values, y_train.values)  # Convert targets to plain arrays

    score = model.score(X_test.values, y_test.values)
    print(f"Fold {fold + 1} R^2 Score: {score:.4f}")

print("\n--- Model Training Complete ---")

# ==========================================
# 3. User Input & Autoregressive Forecasting
# ==========================================
try:
    days_to_predict = int(input("Enter the number of upcoming days to predict: "))
except ValueError:
    print("Please enter a valid integer.")
    exit()

print(f"\nForecasting the next {days_to_predict} market days:")
print("-" * 50)

# Extract starting values safely as clean Python scalars
current_prev_close = float(data["Close"].values[-1])
current_date = pd.to_datetime(data.index[-1])

predicted_days_count = 0
while predicted_days_count < days_to_predict:
    # Advance calendar forward by one day
    current_date += timedelta(days=1)

    # Exclude weekends (Saturday = 5, Sunday = 6)
    if current_date.weekday() >= 5:
        continue

    # Standardize input datatypes to prevent numpy/pandas nesting errors
    future_features = pd.DataFrame(
        [
            {
                "Previous_close": current_prev_close,
                "Day": int(current_date.day),
                "Month": int(current_date.month),
                "Year": int(current_date.year),
            }
        ]
    )

    # Generate prediction (.values handles training array format perfectly)
    prediction = model.predict(future_features.values)[0]

    # Display the result
    date_str = current_date.strftime("%Y-%m-%d")
    print(f"Date: {date_str} | Predicted Close: ${prediction:.2f}")

    # Feed current prediction back into the loop as tomorrow's 'Previous_close'
    current_prev_close = float(prediction)
    predicted_days_count += 1
