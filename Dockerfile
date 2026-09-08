FROM python:3.12-slim

# lightgbm's compiled wheel needs libgomp (OpenMP) at runtime, which isn't
# in the slim base -- without this, `import lightgbm` fails at container
# startup with "libgomp.so.1: cannot open shared object file"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run Docker containers as a non-root user (uid 1000);
# set that up here so files written at runtime (the downloaded weather.db)
# don't hit a permission error. Harmless for a plain local `docker run` too.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# copy just the dependency list first so this layer only rebuilds when
# requirements.txt actually changes, not on every source edit
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --user -r requirements.txt

# now copy the code that's actually needed to serve the API -- no results/,
# no weather.db, both are pulled from Hugging Face at container startup
COPY --chown=user app/ app/
COPY --chown=user src/ src/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
