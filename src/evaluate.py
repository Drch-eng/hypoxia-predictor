import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

MODEL_DIR     = "saved_models"
PROCESSED_DIR = "data/processed"

def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {threshold:.3f}")
    print(f"Precision: {precisions[optimal_idx]:.3f} | Recall: {recalls[optimal_idx]:.3f}")
    return threshold

def run_evaluation():
    model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    y_prob = model.predict(X_test, verbose=0).flatten()
    threshold = find_optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred,
          target_names=["Stable", "Pre-Hypoxia"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Stable", "Pre-Hypoxia"],
                yticklabels=["Stable", "Pre-Hypoxia"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[2].plot(rec, prec, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "evaluation_report.png"), dpi=150)
    print(f"\nEvaluation report saved to {MODEL_DIR}/evaluation_report.png")

if __name__ == "__main__":
    run_evaluation()
