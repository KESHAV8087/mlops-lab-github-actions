import os
import json
from datetime import datetime

import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Read the pointer file to find out which model is the latest.
with open("models/latest_model.json", "r") as f:
    latest = json.load(f)
model_path = latest["model_path"]

# 2. Load that trained model back from disk.
model = joblib.load(model_path)

# 3. Load the exact same held-back test data the training step saved.
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# 4. Use the model to make predictions on the test set.
predictions = model.predict(X_test)

# 5. Compare predictions against the true values using two regression metrics.
#    Mean absolute error: average size of the error, in degrees Celsius.
#    Root mean squared error: similar, but penalizes large misses more.
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

# 6. Save the metrics to a timestamped JSON file in the metrics folder,
#    so each evaluation is recorded and versioned alongside the models.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_path = f"metrics/metrics_{timestamp}.json"
metrics = {
    "model_path": model_path,
    "mean_absolute_error": round(mae, 4),
    "root_mean_squared_error": round(rmse, 4),
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation complete.")
print("Model evaluated:", model_path)
print("Mean absolute error:", round(mae, 4))
print("Root mean squared error:", round(rmse, 4))
print("Saved metrics to:", metrics_path)