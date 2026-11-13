import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/hour_bike_sharing_data.csv")
MODEL_PATH = Path("model.bin")

CAT_COLS = ["season", "mnth", "hr", "weekday", "weathersit"]
NUM_COLS = ["yr", "holiday", "workingday", "temp", "atemp", "hum", "windspeed"]
FEATURE_COLS = CAT_COLS + NUM_COLS


BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def load_and_split(path: Path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["dteday"], format="%d/%m/%y")
    df_clean = df.drop(columns=["instant", "dteday", "casual", "registered"])

    full_train = df_clean[df_clean["date"] < "2012-11-01"].drop(columns=["date"])
    test = df_clean[df_clean["date"] >= "2012-11-01"].drop(columns=["date"])

    X_full_train = full_train[FEATURE_COLS]
    y_full_train_log = np.log1p(full_train["cnt"].values)

    X_test = test[FEATURE_COLS]
    y_test_raw = test["cnt"].values
    return X_full_train, y_full_train_log, X_test, y_test_raw


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    model = LGBMRegressor(**BEST_PARAMS)
    return Pipeline([("prep", preprocessor), ("model", model)])


def main():
    X_train, y_train_log, X_test, y_test_raw = load_and_split(DATA_PATH)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train_log)

    test_preds = np.clip(np.expm1(pipeline.predict(X_test)), 0, None)
    rmse = root_mean_squared_error(y_test_raw, test_preds)
    mae = mean_absolute_error(y_test_raw, test_preds)
    print(f"Final LightGBM | Test RMSE: {rmse:.2f} | Test MAE: {mae:.2f}")

    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(pipeline, f_out)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()