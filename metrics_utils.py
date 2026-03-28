from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["Oui", "Non"])
    report = classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }