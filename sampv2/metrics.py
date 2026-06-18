import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_basic_classifier(y_true, y_pred, y_score=None):
    """Calculate common binary-classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    g_measure = np.sqrt(recall * specificity)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else np.nan

    return pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "MCC",
                "F1 Score",
                "G-Measure",
                "AUC",
                "Precision",
                "Recall (Sensitivity)",
                "Specificity",
                "Jaccard Index",
            ],
            "Value": [
                accuracy,
                mcc,
                f1,
                g_measure,
                auc,
                precision,
                recall,
                specificity,
                jaccard,
            ],
        }
    )
