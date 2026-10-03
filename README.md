# MLOps Lab: GitHub Actions for Model Training and Versioning

This lab automates the training, evaluation, and versioning of a machine learning
model using GitHub Actions. On each trigger, a fresh runner trains a model,
evaluates it, saves timestamped model and metrics files, and commits those
results back into the repository automatically.

## What the pipeline does

1. Loads a real hourly weather dataset for Boston (2024).
2. Trains a Gradient Boosting Regressor to predict temperature from humidity,
   wind speed, cloud cover, hour of day, and month.
3. Evaluates the model on a held-out test set using Mean Absolute Error (MAE)
   and Root Mean Squared Error (RMSE).
4. Saves a timestamped model file to `models/` and a timestamped metrics file
   to `metrics/`, then commits them back to the repository.

## Automation (GitHub Actions)

The workflow in `.github/workflows/train.yml` runs:

- On every push to `main`
- On a daily schedule (midnight UTC)
- Manually, via the "Run workflow" button

## My modifications to the original lab

- Replaced the original synthetic dataset with real Boston weather data pulled
  from the free Open-Meteo historical API.
- Replaced the original classifier with a Gradient Boosting Regressor, turning
  the task into a regression problem.
- Replaced the classification metric with regression metrics (MAE and RMSE).
- Added folder-creation logic so the scripts run correctly on a clean runner.

## Files

- `fetch_data.py` - fetches the real weather dataset into `data/`
- `train_model.py` - trains and saves a timestamped model
- `evaluate_model.py` - evaluates the latest model and records metrics
- `.github/workflows/train.yml` - the automation workflow