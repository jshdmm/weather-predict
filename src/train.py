# Packages
import pandas as pd
import numpy as np
import sys
sys.path.append("..")   # repo root
from src.setup_db import weatherDB
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path


db_path = Path.cwd().parent / "weather.db"


def train_model(db_path, period, seed) -> pd.DataFrame:

    db = weatherDB(f"sqlite:///{db_path}")
    df = db.get_weather_data()

    return df


df = train_model(db_path = db_path, period = 730, seed = 4036018)

print(df.head())





