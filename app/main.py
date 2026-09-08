from fastapi import FastAPI
from src.setup_db import weatherDB
from src.features import FEATURE_COLS, add_time_features
from src.upload_model import download_latest_model, download_latest_results, DEFAULT_REPO_ID
from src.sync_db import download_weather_db

app = FastAPI()

DB_URL = "sqlite:///weather.db"

# pull the shared weather.db (written by the retrain cronjob) once at startup
download_weather_db()



# create health route to check if the API is running
@app.get("/health")
def read_health():
    return {"status": "ok"}

# create root route to welcome users
@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API!"}

# load the latest trained model from Hugging Face Hub
model = download_latest_model()
MODEL_PATH = f"hf://{DEFAULT_REPO_ID}/model.pkl"

# load the results summary uploaded alongside the model
results = download_latest_results()

# check model path
@app.get("/model_path")
def read_model_path():
    return {"model_path": MODEL_PATH}

# entry for predict route
@app.get("/predict/")
def predict():
    return("Please select predict/historical for a historical model prediction for Open Meteo archive weather data and predict/forecast for a weather forecast for the upcoming days.")

# prediction for the latest known archive row (historical data), i.e. as of
# the day the cronjob last retrained -- rows only have a real
# temperature_2m once the archive fetch has confirmed them, so filtering on
# that (rather than just taking the last row by time) skips over the
# forecast-only rows that extend further into the future
@app.get("/predict/historical")
def predict_historical():
    db = weatherDB(DB_URL)
    df = db.get_weather_data().sort_values("time")

    historical = df[df["temperature_2m"].notna()]
    latest = historical.iloc[[-1]]
    latest = add_time_features(latest)

    # get prediction
    pred = model.predict(latest[FEATURE_COLS])[0]

    return {
        "model_path": MODEL_PATH,
        "time": latest["time"].iloc[0].isoformat(),
        "predicted_temp_model": round(float(pred), 2),
        "predicted_temp_open_meteo": round(float(latest["temperature_2m"].iloc[0]), 2),
        "results": results,
    }

# predictions for upcoming days (forecast), starting right where historical
# data ends -- the same day-of-retraining boundary as /predict/historical,
# instead of the server's wall-clock time, so both endpoints agree
@app.get("/predict/forecast")
def predict_forecast():
    db = weatherDB(DB_URL)
    df = db.get_weather_data().sort_values("time")

    last_known_time = df.loc[df["temperature_2m"].notna(), "time"].max()
    upcoming = df[df["predicted_temperature_2m"].notna() & (df["time"] > last_known_time)]

    return [
        {
            "time": row["time"].isoformat(),
            "predicted_temperature_2m": round(float(row["predicted_temperature_2m"]), 2),
        }
        for _, row in upcoming.iterrows()
    ]

