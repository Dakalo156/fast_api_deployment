from pathlib import Path
import yaml
import pickle
import re

# Get parent directory for this file
BASE_DIR = Path(__file__).resolve(strict=True).parent
# print(BASE_DIR)

# Load the language classes list from yaml file
with open(f"{BASE_DIR}\language_classes.yml", "rb") as f:
    language_classes = yaml.safe_load(f)

LANGUAGE_CLASSES = language_classes["LANGUAGE_CLASSES"]
MODEL_VERSION = language_classes["MODEL_VERSION"][0]
# print(LANGUAGE_CLASSES)
# print(MODEL_VERSION)

# Load the trained Model
with open(f"{BASE_DIR}/trained_language_detection-{MODEL_VERSION}.pkl", "rb") as f:
    model = pickle.load(f)


# Make Predictions
def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    language = LANGUAGE_CLASSES[pred[0]]
    return language
