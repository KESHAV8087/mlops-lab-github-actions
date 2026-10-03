import json
import urllib.request
import pandas as pd

# Open-Meteo historical weather API for Boston, year 2024.
# Free, no API key required.
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=42.36&longitude=-71.06"
    "&start_date=2024-01-01&end_date=2024-12-31"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover"
)

# Download the data and parse the JSON response.
with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())

# The hourly readings come back as lists under the "hourly" key.
# Putting that dictionary into a DataFrame lines them up into columns.
hourly = data["hourly"]
df = pd.DataFrame(hourly)

# Save to the data folder as a CSV, without the extra index column.
df.to_csv("data/weather.csv", index=False)

# Print a quick summary so we can verify it worked.
print("Shape:", df.shape)
print(df.head())