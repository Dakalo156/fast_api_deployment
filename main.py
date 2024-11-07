from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from pathlib import Path
import yaml


# Get parent directory for this file
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the language classes list from yaml file
with open(f"{BASE_DIR}\\app\model\language_classes.yml", "rb") as f:
    language_classes = yaml.safe_load(f)

MODEL_VERSION = language_classes["MODEL_VERSION"][0]

# Initiate FastAPI
app = FastAPI(title="Language Detection Model")


# type hints for input text
class TextInput(BaseModel):
    text: str


# type hints for model prediction
class TextOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=TextOut)
def predict(payload: TextInput):
    language = predict_pipeline(payload.text)
    return {"language": language}
