# Weather Predict — Berlin/Tempelhof

This project was created as an extension of my [ELT pipepline project](https://github.com/phipsrick/GA2024_1) which covered integrating weather data with different IoT sensors from a Dutch household. I wanted to learn using CI/CID with GitHub Actions and deploying a simple ensemble-based model on Hugging Face, containerized with Docker and served through FastAPI. Besides reading through docs and tutorials, I used Claude Code as an assistant to work through the project. In doing so, I was still making relevant decisions myself (also see the commit history to follow along the project trajectory).

An end-to-end **MLOps** project: a LightGBM model that predicts temperature for Berlin/Tempelhof, retrained daily on fresh weather data, served through a containerized FastAPI API. The focus here isn't the model itself (a gradient-boosted tree on a handful of weather features) — it's the pipeline around it: automated retraining, a model/data registry, reproducible reporting, and stateless serving.

## Why this project

Built to practice the operational side of ML that tutorials usually skip: what happens *after* `model.fit()`. Specifically — how do you keep a model current without manual intervention, where does "the current model" actually live, how do multiple environments (a laptop, a CI runner, a container) agree on the same data and model version, and how do you serve predictions without silently depending on whatever happens to be on one developer's disk.

## Architecture

```
Open-Meteo Archive API ──┐
                          ├──> weather.db (SQLite) <──> Hugging Face Dataset repo
Open-Meteo Forecast API ─┘         │                    (shared, synced pull/push)
                                    ▼
                          LightGBM training (src/train.py)
                                    │
                         ┌──────────┴──────────┐
                         ▼                      ▼
                  model.pkl + results.json   forecast predictions
                  → Hugging Face Model repo    written back into weather.db
                         │
                         ▼
              FastAPI app (Docker) — downloads model, results,
              and weather.db from Hugging Face at startup
                         │
                         ▼
              /predict/historical   /predict/forecast
```

Everything the API needs — the trained model, its evaluation metrics, and the database — is pulled from Hugging Face at container startup. The image itself carries no local model file, no local database, and no training artifacts; verified by running it in a directory containing nothing but the application code.

## MLOps components

- **Automated daily retraining** — a GitHub Actions cron (`retrain.yml`, 04:00 UTC) fetches a rolling 730-day archive window, retrains the model, and generates next-week's forecast predictions in one run, with `workflow_dispatch` for on-demand runs.
- **Model registry** — every retrain uploads `model.pkl` and a structured `results.json` (model type, training period, date range, MAE) to a public [Hugging Face model repo](https://huggingface.co/jshdmm/weather-predict-berlin), giving every model version a URL and a history.
- **Shared data store** — `weather.db` is synced to/from a [Hugging Face dataset repo](https://huggingface.co/datasets/jshdmm/weather-predict-berlin-db) (`src/sync_db.py`) rather than living only on one machine, so the cronjob, local development, and the served API all read the same data.
- **Separation of CI from production data** — `ci.yml` runs lint/tests/a training smoke test on every push; only `retrain.yml` (the actual cron) pulls and pushes the shared dataset, so pushing code can never clobber production data with an in-progress branch.
- **Structured, reproducible reporting** — training results are saved as JSON (not just a log line), consumed identically by the CI PR comment (via [CML](https://cml.dev/)) and the API response.
- **Stateless, containerized serving** — a non-root Docker image (Hugging Face Spaces-compatible: uid 1000, `libgomp1` for LightGBM) with no baked-in data; verified locally end-to-end (build → run → hit every endpoint) before being treated as done.
- **Two prediction modes, one consistent boundary** — `/predict/historical` and `/predict/forecast` split the data at "the last timestamp with a confirmed observation," so both endpoints agree on what "now" means regardless of server wall-clock time.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /predict/historical` | Model's prediction for the latest confirmed archive row, alongside Open-Meteo's own value for the same timestamp and the model's training metrics |
| `GET /predict/forecast` | Model's predictions for the upcoming week, derived from Open-Meteo's forecast features |
| `GET /model_path` | Which model artifact is currently loaded |

## Running it

**Locally:**
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Via Docker:**
```bash
docker build -t weather-predict-api .
docker run -p 8000:8000 weather-predict-api
```
No credentials needed to run the API — the model and database repos are public reads. A `HF_TOKEN` (write-scoped) is only required for the training/retraining side, which writes back to Hugging Face.

**Training/retraining manually:**
```bash
make train      # fetch archive + forecast data, retrain, store forecast predictions
make upload      # push model.pkl + results.json to the model repo
make db-push     # push the updated weather.db to the dataset repo
```

## Tech stack

Python · FastAPI · LightGBM · SQLAlchemy (SQLite) · pandas · Docker · GitHub Actions · Hugging Face Hub (model + dataset repos) · CML

## Data

[Open-Meteo](https://open-meteo.com/) — the archive (ERA5 reanalysis) API for training data, and the forecast API for next-week features — for Berlin/Tempelhof (52.4676, 13.4020).

## Testing

`pytest` covers database setup and upsert behavior (`tests/`); `ruff` enforces lint on every push via CI.

## Roadmap

- Model monitoring: compare `/predict/forecast`'s stored predictions against the real observed values once the archive catches up to them, to measure actual forecast accuracy over time (not just offline test-set MAE).
- Structured logging for prediction requests.
- Pruning `weather.db` to enforce the intended rolling window (currently archive rows accumulate rather than age out).
