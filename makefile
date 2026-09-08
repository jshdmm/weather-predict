DBURL ?= sqlite:///weather.db
PERIOD ?= 730
MODEL_REPO_ID ?= jshdmm/weather-predict-berlin

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python -m src.main --dburl $(DBURL) --period $(PERIOD)

upload:
	python -m src.upload_model --repo-id $(MODEL_REPO_ID)

db-pull:
	python -m src.sync_db --action pull

db-push:
	python -m src.sync_db --action push

results:
	@latest_json=$$(ls -t results/model_*.json | head -n 1); \
	latest_png=$$(ls -t results/model_*.png | head -n 1); \
	echo "## Model Metrics" > results.md; \
	echo "" >> results.md; \
	python -c "import json; print(json.load(open('$$latest_json'))['summary'])" >> results.md; \
	echo "" >> results.md; \
	echo "## Test Evaluation Plot" >> results.md; \
	echo "![Test Evaluation]($$latest_png)" >> results.md; \
	cml comment create results.md

.PHONY: install format train upload db-pull db-push results
