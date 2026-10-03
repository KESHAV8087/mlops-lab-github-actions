import os
import json
from datetime import datetime

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Make sure the output folders exist on any machine, including a fresh runner.
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 1. Load the real weather data we fetched and committed earlier.
df = pd.read_csv("data/weather.csv")

# 2. Turn the time column into useful numeric features.
#    A model cannot read a timestamp string, so we extract hour and month,
#    which carry the daily and seasonal patterns in temperature.
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month

# 3. Define what we predict (y) and what we predict it from (X).
#    Target: temperature. Features: the other weather values plus time features.
feature_columns = ["relative_humidity_2m", "wind_speed_10m", "cloud_cover", "hour", "month"]
X = df[feature_columns]
y = df["temperature_2m"]

# 4. Split into a training set and a test set.
#    The model learns on the training set; the test set is held back
#    so we can later measure how it does on data it has not seen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create and train the gradient boosting regressor.
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. Save a timestamped version of the trained model into the models folder.
#    The timestamp in the filename is how this lab does model versioning:
#    every run produces a new, uniquely named model file.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/model_{timestamp}.joblib"
joblib.dump(model, model_path)

# 7. Also save the test set so the evaluation step can reuse the exact
#    same held-back data, keeping training and evaluation consistent.
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# 8. Record which model file is the latest, so evaluation knows what to load.
with open("models/latest_model.json", "w") as f:
    json.dump({"model_path": model_path}, f)

print("Training complete.")
print("Saved model to:", model_path)
print("Training rows:", len(X_train), "Test rows:", len(X_test))