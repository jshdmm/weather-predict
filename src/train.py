# Packages
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.setup_db import weatherDB
from src.fetch_weather import get_archive
import pickle


def train_model(dburl: str, period, seed) -> pd.DataFrame:

    # get database and data
    db = weatherDB(dburl)
    df = db.get_weather_data()

    # get period, start, end date, and corresponding archive URL
    archive_info = get_archive(period)

    

    # sort by time index for sorted training indices (no future leakage)
    df = df.sort_values("time").reset_index(drop=True)

    # get time features, sin cos since time is cyclical
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # all features
    feature_cols = [
        "relativehumidity_2m", "rain", "snowfall", "windspeed_10m",
        "winddirection_10m", "soil_temperature_0_to_7cm",
        "hour_sin", "hour_cos", "dayofyear",
    ]

    # features, target
    X = df[feature_cols]
    y = df["temperature_2m"]

    # time-ordered split to avoid future leakage (85% training data)
    split = int(len(df) * 0.85)
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
        "period": period,
        "start_date": archive_info["start_date"],
        "end_date": archive_info["end_date"],
        "X_test": X_test,
        "y_test": y_test,
    }

    # save results
    with open (f"results/model_{archive_info['start_date']}_{archive_info['end_date']}.pkl", "wb") as f:
        pickle.dump(model, f)

    return results

