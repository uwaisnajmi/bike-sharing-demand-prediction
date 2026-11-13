import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_PATH = Path("model.bin")
FEATURE_COLS = [
    "season",
    "mnth",
    "hr",
    "weekday",
    "weathersit",
    "yr",
    "holiday",
    "workingday",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


def predict_demand(record: dict[str, Any], model=None) -> float:
    if model is None:
        model = load_model()
    df_input = pd.DataFrame([record])[FEATURE_COLS]
    pred_log = model.predict(df_input)[0]
    pred_cnt = float(np.clip(np.expm1(pred_log), 0.0, None))
    return round(pred_cnt, 2)


if __name__ == "__main__":
    sample_hour = {
        "season": 3,
        "yr": 1,
        "mnth": 8,
        "hr": 17,
        "holiday": 0,
        "weekday": 4,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.76,
        "atemp": 0.70,
        "hum": 0.50,
        "windspeed": 0.19,
    }
    prediction = predict_demand(sample_hour)
    print(f"Predicted hourly bike rentals: {prediction}")