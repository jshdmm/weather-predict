# Packages
import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from src.setup_db import weatherDB
from src.features import FEATURE_COLS, add_time_features
import matplotlib.pyplot as plt

def train_model(dburl: str, archive_dict: dict, seed: int = 4036018) -> dict:

    # create results directory on fresh run (hidden in .gitignore otherwise)
    os.makedirs("results", exist_ok=True)

    # get database and data
    db = weatherDB(dburl)
    df = db.get_weather_data()
    

    # sort by time index for sorted training indices (no future leakage)
    df = df.sort_values("time").reset_index(drop=True)

    # get time features, sin cos since time is cyclical
    df = add_time_features(df)

    # get archive data based on the provided start and end dates
    df_archive = df[df["time"].dt.date.between(archive_dict["start_date"], archive_dict["end_date"])].copy()

    # features, target
    X = df_archive[FEATURE_COLS]
    y = df_archive["temperature_2m"]

    # time-ordered split to avoid future leakage (85% training data)
    split = int(len(df_archive) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # train the model
    model = lgb.LGBMRegressor(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))


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

    # save model
    model_path = f"results/model_{archive_dict['start_date']}_{archive_dict['end_date']}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}.")

    # save results as a structured summary (used by the API and CI report)
    summary_text = f"LightGBM Model trained on data from {archive_dict['start_date']} to {archive_dict['end_date']} in a period of {archive_dict['period']} days with a test MAE of {mae:.2f} °C."
    results_summary = {
        "model_type": "LightGBM",
        "period_days": archive_dict["period"],
        "start_date": str(archive_dict["start_date"]),
        "end_date": str(archive_dict["end_date"]),
        "mae": round(float(mae), 2),
        "summary": summary_text,
    }
    with open(f"results/model_{archive_dict['start_date']}_{archive_dict['end_date']}.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(summary_text)





    # get a test performance plot for evaluation
    # Predicted vs. Open-Meteo Prediction over test period
    test_time = df_archive["time"].iloc[split:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_time, y_test, color="#2a78d6", linewidth=1.5, label="Open-Meteo")
    ax.plot(test_time, preds, color="#eb6834", linewidth=1.5, label="LightGBM")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Open-Meteo Prediction vs. LightGBM temperature prediction (test period)")
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    with open(f"results/model_{archive_dict['start_date']}_{archive_dict['end_date']}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300)
    print("Saved test performance plot.")




    return results

