from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

LABELS = ["H", "D", "A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DNN for 1X2 football match outcome prediction.")
    parser.add_argument("--data-path", default="data/processed/training_dataset.parquet")
    parser.add_argument("--features-path", default="data/processed/feature_columns.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(input_dim: int, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = sorted(df["season_start_year"].unique())
    if len(seasons) < 3:
        raise ValueError("At least three seasons are required for train/validation/test split.")
    return (
        df[df["season_start_year"].isin(seasons[:-2])].copy(),
        df[df["season_start_year"] == seasons[-2]].copy(),
        df[df["season_start_year"] == seasons[-1]].copy(),
    )


def predict_percentages(model: tf.keras.Model, scaler: StandardScaler, frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    probabilities = model.predict(scaler.transform(frame[feature_columns]), verbose=0)
    prediction = pd.DataFrame(probabilities, columns=["home_win_pct", "draw_pct", "away_win_pct"])
    prediction["predicted_label"] = prediction[["home_win_pct", "draw_pct", "away_win_pct"]].idxmax(axis=1)
    prediction["predicted_label"] = prediction["predicted_label"].map(
        {"home_win_pct": "1", "draw_pct": "X", "away_win_pct": "2"}
    )
    return prediction


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.data_path)
    feature_columns = json.loads(Path(args.features_path).read_text(encoding="utf-8"))
    train_df, val_df, test_df = temporal_split(dataset)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_columns])
    x_val = scaler.transform(val_df[feature_columns])
    x_test = scaler.transform(test_df[feature_columns])
    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    model = build_model(len(feature_columns), args.seed)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
        ],
    )

    test_prob = model.predict(x_test, verbose=0)
    test_pred = np.argmax(test_prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "log_loss": float(log_loss(y_test, test_prob)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "classification_report": classification_report(
            y_test,
            test_pred,
            labels=[0, 1, 2],
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    scaler_path = model_dir / "feature_scaler.joblib"
    model_path = model_dir / "football_outcome_dnn.keras"
    metrics_path = model_dir / "metrics.json"
    history_path = model_dir / "history.json"
    sample_predictions_path = model_dir / "test_predictions.csv"

    joblib.dump({"scaler": scaler, "feature_columns": feature_columns}, scaler_path)
    model.save(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    predictions = test_df[["match_date", "league_name", "home_team", "away_team", "result_code"]].reset_index(drop=True)
    predictions = pd.concat([predictions, predict_percentages(model, scaler, test_df, feature_columns)], axis=1)
    predictions.to_csv(sample_predictions_path, index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()
