from sklearn.metrics import balanced_accuracy_score, recall_score, specificity_score, roc_auc_score, matthews_corrcoef
from sklearn.base import BaseEstimator
import numpy as np

def evaluate(model: BaseEstimator, X: np.ndarray, y: np.ndarray, average='weighted'):
    # try:
    y_true, y_pred, pred_prob = y, model.predict(X), model.predict_proba(X)
    y_prob = net.predict_proba(X)

    metrics = {
        'BA': balanced_accuracy_score(y_true, y_pred)
        'SEN': recall_score(y_true, y_pred, average=average)
        'SPE': specificity_score(y_true, y_pred, average=average)
        'AUC': roc_auc_score(y_true, pred_prob, multi_class="ovr", average=average)
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    return metrics

def evaluate_voting(y_true, y_pred, pred_prob, average='weighted'):
    metrics = {
        'BA': balanced_accuracy_score(y_true, y_pred),
        'SEN': recall_score(y_true, y_pred, average=average),
        'SPE': specificity_score(y_true, y_pred, average=average),
        'AUC': roc_auc_score(y_true, pred_prob, multi_class="ovr", average=average),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    return metrics
