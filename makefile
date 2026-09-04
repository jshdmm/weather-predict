DBURL ?= sqlite:///weather.db
PERIOD ?= 730

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python -m src.main --dburl $(DBURL) --period $(PERIOD)

results:
	@latest_txt=$$(ls -t results/model_*.txt | head -n 1); \
	latest_png=$$(ls -t results/model_*.png | head -n 1); \
	echo "## Model Metrics" > results.md; \
	echo "" >> results.md; \
	cat $$latest_txt >> results.md; \
	echo "" >> results.md; \
	echo "## Test Evaluation Plot" >> results.md; \
	echo "![Test Evaluation]($$latest_png)" >> results.md; \
	cml comment create results.md

.PHONY: install format train results
