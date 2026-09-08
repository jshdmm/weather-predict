from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    model_type: str
    version: float
    results: dict | None = None


# create health route to check if the API is running
@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API!"}

