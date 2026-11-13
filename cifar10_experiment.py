"""Compare three CIFAR-10 CNN recipes without using the test set for selection."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


MODEL_CONFIGS = {
    "relu_cnn": {"epochs": 50, "batch_size": 32, "optimizer": "rmsprop"},
    "batchnorm_cnn": {"epochs": 100, "batch_size": 500, "optimizer": "adamax_fast"},
    "tanh_cnn": {"epochs": 150, "batch_size": 32, "optimizer": "adamax_slow"},
}


def _load_cifar_batch(path: Path):
    """Load one CIFAR-10 batch from the local Python-format dataset."""
    with path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")

    data = batch[b"data"]
    labels = np.asarray(batch[b"labels"], dtype="int32")

    if data.shape != (10000, 3072):
        raise ValueError(f"Unexpected image shape in {path.name}: {data.shape}")

    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    if len(images) != len(labels):
        raise ValueError(f"Image/label mismatch in {path.name}")

    return images, labels


def prepare_data(seed: int, smoke: bool):
    """Load local CIFAR-10 and return stratified train/validation data."""
    keras_home = Path(
        os.environ.get("KERAS_HOME", str(Path.home() / ".keras"))
    )
    dataset_dir = keras_home / "datasets" / "cifar-10-batches-py"

    train_files = [
        dataset_dir / f"data_batch_{index}"
        for index in range(1, 6)
    ]
    test_file = dataset_dir / "test_batch"

    required_files = train_files + [test_file]
    missing = [str(path) for path in required_files if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Local CIFAR-10 files are missing:\n"
            + "\n".join(missing)
        )

    # Preallocate arrays to avoid holding duplicate training batches in memory.
    x_full = np.empty((50000, 32, 32, 3), dtype=np.uint8)
    y_full = np.empty(50000, dtype=np.int32)

    offset = 0

    for batch_path in train_files:
        batch_x, batch_y = _load_cifar_batch(batch_path)
        end = offset + len(batch_y)

        if end > 50000:
            raise ValueError("Unexpected number of training samples.")

        x_full[offset:end] = batch_x
        y_full[offset:end] = batch_y
        offset = end

        del batch_x, batch_y

    if offset != 50000:
        raise ValueError(f"Expected 50000 training samples, got {offset}.")

    x_test, y_test = _load_cifar_batch(test_file)

    if x_test.shape != (10000, 32, 32, 3):
        raise ValueError(f"Unexpected test shape: {x_test.shape}")

    y_full = y_full.ravel().astype("int32")
    y_test = y_test.ravel().astype("int32")

    train_idx, val_idx = train_test_split(
        np.arange(len(y_full)),
        test_size=0.10,
        random_state=seed,
        stratify=y_full,
    )

    if smoke:
        train_idx, _ = train_test_split(
            train_idx,
            train_size=3000,
            random_state=seed,
            stratify=y_full[train_idx],
        )

        val_idx, _ = train_test_split(
            val_idx,
            train_size=500,
            random_state=seed,
            stratify=y_full[val_idx],
        )

    x_train = x_full[train_idx].astype("float32") / 255.0
    x_val = x_full[val_idx].astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (
        (x_train, y_full[train_idx]),
        (x_val, y_full[val_idx]),
        (x_test, y_test),
    )


def build_model(name: str) -> tf.keras.Model:
    """Recreate the three broad CNN architectures of the source notebook."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = inputs

    if name == "relu_cnn":
        for filters in (32, 32, 32, 48, 48):
            x = tf.keras.layers.Conv2D(
                filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
            )(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dense(500, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    elif name == "batchnorm_cnn":
        for filters in (32, 32, 32, 48, 48, 48, 64, 128):
            x = tf.keras.layers.Conv2D(
                filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-3)

    else:
        for filters in (32, 32, 32, 48, 48, 48, 48):
            x = tf.keras.layers.Conv2D(
                filters, 3, padding="same", activation="tanh",
                kernel_initializer="glorot_uniform"
            )(x)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dense(
            500, activation="tanh", kernel_initializer="glorot_uniform"
        )(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        optimizer = tf.keras.optimizers.Adamax(learning_rate=1e-4)

    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history: dict, destination: Path) -> None:
    """Persist a comparison of training and validation learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history[key], label="Train")
        ax.plot(history[f"val_{key}"], label="Validation")
        ax.set(xlabel="Epoch", ylabel=key.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_experiments(
    *,
    smoke: bool = False,
    epochs: tuple[int, int, int] = (50, 100, 150),
    seed: int = 42,
) -> Path:
    """Train all candidates; evaluate the test set only after full-run selection."""
    if len(epochs) != 3 or any(value < 1 for value in epochs):
        raise ValueError("Supply exactly three positive epoch budgets.")

    tf.keras.utils.set_random_seed(seed)
    train, val, test = prepare_data(seed, smoke)
    destination = Path("results") / ("smoke" if smoke else "full")
    destination.mkdir(parents=True, exist_ok=True)
    summary = []

    for (name, config), max_epochs in zip(MODEL_CONFIGS.items(), epochs):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(name)
        checkpoint = destination / f"{name}.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", mode="min", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=10, restore_best_weights=True
            ),
        ]
        history_obj = model.fit(
            train[0], train[1],
            validation_data=val,
            batch_size=config["batch_size"],
            epochs=1 if smoke else max_epochs,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        history = {key: [float(x) for x in values]
                   for key, values in history_obj.history.items()}
        best_idx = int(np.argmin(history["val_loss"]))
        row = {
            "model": name,
            "epochs_trained": len(history["loss"]),
            "best_epoch": best_idx + 1,
            "best_val_loss": history["val_loss"][best_idx],
            "best_val_accuracy": history["val_accuracy"][best_idx],
            "batch_size": config["batch_size"],
            "checkpoint": str(checkpoint),
        }
        summary.append(row)
        write_json(destination / f"{name}_history.json", history)
        save_history_plot(history, destination / f"{name}_curves.png")
        del model
        gc.collect()

    with (destination / "validation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    selected = min(summary, key=lambda row: row["best_val_loss"])
    write_json(destination / "selected_model.json", selected)
    print(f"Selected model using validation loss: {selected['model']}")

    if smoke:
        print("Smoke run: official test set NOT evaluated.")
        return destination

    # Only the validation-selected checkpoint may access the official test set.
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(selected["checkpoint"])
    test_loss, test_accuracy = model.evaluate(test[0], test[1], verbose=0)
    predictions = model.predict(test[0], batch_size=256, verbose=0).argmax(axis=1)
    report = classification_report(
        test[1], predictions, labels=list(range(10)),
        output_dict=True, zero_division=0,
    )
    metrics = {
        "selected_model": selected["model"],
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy_score(test[1], predictions)),
        "keras_test_accuracy": float(test_accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
    }
    write_json(destination / "test_metrics.json", metrics)

    matrix = confusion_matrix(test[1], predictions, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    print(f"Official held-out test accuracy: {metrics['test_accuracy']:.4f}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Short run; skip test evaluation")
    parser.add_argument(
        "--epochs", nargs=3, type=int, default=[50, 100, 150],
        metavar=("RELU", "BATCHNORM", "TANH"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    destination = run_experiments(
        smoke=args.smoke, epochs=tuple(args.epochs), seed=args.seed
    )
    print(f"Reports saved to: {destination.resolve()}")


if __name__ == "__main__":
    main()
