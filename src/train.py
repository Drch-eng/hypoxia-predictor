import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from model import build_lstm_model

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "saved_models"
BATCH_SIZE    = 256
EPOCHS        = 100
LEARNING_RATE = 1e-3
RANDOM_SEED   = 42

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_data():
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"Class weights: {cw}")
    return cw

def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
    ]

def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history.history["loss"],      label="Train")
    axes[0].plot(history.history["val_loss"],  label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["auc"],       label="Train")
    axes[1].plot(history.history["val_auc"],   label="Val")
    axes[1].set_title("AUC"); axes[1].legend()
    axes[2].plot(history.history["precision"], label="Precision")
    axes[2].plot(history.history["recall"],    label="Recall")
    axes[2].set_title("Precision / Recall"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print("Training curves saved.")

def main():
    X, y = load_data()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.90,
        stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=[128, 64],
        dropout_rate=0.3,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=get_class_weights(y_train),
        callbacks=build_callbacks(),
        verbose=1
    )

    print("\n── Final Test Evaluation ──")
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    plot_history(history)
    print(f"\nModel saved to {MODEL_DIR}/best_model.keras")

if __name__ == "__main__":
    main()
