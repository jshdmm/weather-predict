# Packages
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.setup_db import weatherDB
from src.fetch_weather import get_archive
import pickle
import datetime as dt

def train_model(dburl: str, archive_dict: dict, seed) -> pd.DataFrame:

    # get database and data
    db = weatherDB(dburl)
    df = db.get_weather_data()
    

    # sort by time index for sorted training indices (no future leakage)
    df = df.sort_values("time").reset_index(drop=True)

    # get time features, sin cos since time is cyclical
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # get archive data based on the provided start and end dates
    print(archive_dict["start_date"], archive_dict["end_date"])
    df_archive = df[df["time"].dt.date.between(archive_dict["start_date"], archive_dict["end_date"])].copy()


    # all features
    feature_cols = [
        "relativehumidity_2m", "rain", "snowfall", "windspeed_10m",
        "winddirection_10m", "soil_temperature_0_to_7cm",
        "hour_sin", "hour_cos", "dayofyear",
    ]

    # features, target
    X = df_archive[feature_cols]
    y = df_archive["temperature_2m"]

    # time-ordered split to avoid future leakage (85% training data)
    split = int(len(df_archive) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # train the model
    model = lgb.LGBMRegressor(n_estimators=200, random_state=125)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    print(f"Test MAE: {mae:.2f} °C")

    # save model results as dict
    results = {
        "mae": mae,
        "model": model,
        "period": archive_dict["period"],
        "start_date": archive_dict["start_date"],
        "end_date": archive_dict["end_date"],
        "X_test": X_test,
        "y_test": y_test,
    }

    # save results
    with open (f"results/model_{archive_dict['start_date']}_{archive_dict['end_date']}.txt", "w") as f:
        f.write(f"LightGBM Model trained on data from {archive_dict['start_date']} to {archive_dict['end_date']} in a period of {archive_dict['period']} days was trained with a test MAE of {mae:.2f} °C.\n")

    return results

