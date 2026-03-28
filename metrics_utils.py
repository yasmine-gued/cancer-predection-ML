from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(y_true, y_pred):
    labels = ["Non", "Oui"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }