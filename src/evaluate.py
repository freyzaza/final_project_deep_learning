import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_confusion_heatmap(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_training_curves(history, model_name, save_dir):
    ensure_dir(save_dir)

    # Accuracy curve
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{model_name}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{model_name}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_model(model_name, y_true, y_pred, labels, log_dir="outputs/logs/"):
    ensure_dir(log_dir)

    # -------- Accuracy --------
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {model_name} ACCURACY: {acc:.4f} ===\n")

    # -------- Classification Report --------
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

    # Save classification report
    report_path = f"{log_dir}/{model_name}_classification_report.csv"
    df_report.to_csv(report_path)
    print(f"[SAVED] Classification Report → {report_path}")

    # -------- Heatmap --------
    heatmap_path = f"{log_dir}/{model_name}_heatmap.png"
    plot_confusion_heatmap(
        y_true, y_pred, labels,
        title=f"Confusion Matrix - {model_name}",
        save_path=heatmap_path
    )
    print(f"[SAVED] Heatmap → {heatmap_path}")

    return acc, df_report


def full_evaluation(model_name, history, y_true, y_pred, labels):
    """
    Evaluates:
    - Training curves
    - Classification report
    - Confusion matrix heatmap
    """
    log_dir = "outputs/logs/"
    ensure_dir(log_dir)

    # Training curves
    plot_training_curves(history, model_name, log_dir)

    # Accuracy + report + heatmap
    acc, report = evaluate_model(model_name, y_true, y_pred, labels, log_dir)

    print(f"\n==== Evaluation for {model_name} DONE ====\n")
    return acc, report
