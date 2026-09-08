import glob
from fastapi import FastAPI
from pydantic import BaseModel
from src.setup_db import weatherDB
from src.features import FEATURE_COLS, add_time_features
from src.upload_model import download_latest_model, DEFAULT_REPO_ID

app = FastAPI()

DB_URL = "sqlite:///weather.db"



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

# get results path
RESULTS_PATH = sorted(glob.glob("results/model*.txt"))[-1]

# read results
with open(RESULTS_PATH, "r") as f:
    results = f.read()

# check model path
@app.get("/model_path")
def read_model_path():
    return {"model_path": MODEL_PATH}

# predict endpoint to get the latest weather prediction
@app.get("/predict")
def predict():
    db = weatherDB(DB_URL)
    df = db.get_weather_data().sort_values("time")

    # get latest row and add time features
    latest = df.iloc[[-1]]
    latest = add_time_features(latest)

    # get prediction
    pred = model.predict(latest[FEATURE_COLS])[0]

    return {
        "model_path": MODEL_PATH,
        "time": latest["time"].iloc[0].isoformat(),
        "predicted_temp_model": round(float(pred), 2),
        "predicted_temp_open_meteo": round(float(df["temperature_2m"].iloc[0]), 2),
        "results": results,
    }

