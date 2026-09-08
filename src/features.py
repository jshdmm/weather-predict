import numpy as np
import pandas as pd

# feature columns used for training and prediction
FEATURE_COLS = [
    "relativehumidity_2m", "rain", "snowfall", "windspeed_10m",
    "winddirection_10m",
    "hour_sin", "hour_cos", "dayofyear",
]

# Convert and add time features
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24) # sin transformation for cyclical hour feature
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24) # cos 
    return df
