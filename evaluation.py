import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

def _predict_torch(model, data_loader, device):
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch.to(device).float())
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_scores.extend(probs)
            y_true.extend(y_batch.numpy())
    return np.asarray(y_true), np.asarray(y_scores)

def _compute_metrics(y_true, y_scores, threshold, include_curves=True):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "roc_auc": auc(fpr, tpr),
        "auprc": average_precision_score(y_true, y_scores),
        "threshold": threshold,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, output_dict=False, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if include_curves:
        metrics.update({
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
        })
    return metrics, fpr, tpr, recall_curve, precision_curve

def evaluate_model(model, data_loader, device, threshold=None, target_sens=0.9):
    y_true, y_scores = _predict_torch(model, data_loader, device)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    if threshold is None:
        idx = np.argmin(np.abs(tpr - target_sens))
        threshold = roc_thresholds[idx]
    return _compute_metrics(y_true, y_scores, threshold)

def evaluate_model_subject(model, data_loader, device, subject_ids, threshold=None, target_sens=0.9):
    y_true, y_scores = _predict_torch(model, data_loader, device)
    subject_ids = np.asarray(subject_ids)
    if len(subject_ids) != len(y_scores):
        raise ValueError("subject_ids must match the number of samples in data_loader.")

    sub_true, sub_scores = [], []
    for sid in np.unique(subject_ids):
        idx = subject_ids == sid
        labels = y_true[idx]
        if not np.all(labels == labels[0]):
            raise ValueError(f"Subject {sid} contains inconsistent labels.")
        sub_true.append(labels[0])
        sub_scores.append(np.mean(y_scores[idx]))

    y_true = np.asarray(sub_true)
    y_scores = np.asarray(sub_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    if threshold is None:
        idx = np.argmin(np.abs(tpr - target_sens))
        threshold = roc_thresholds[idx]
    return _compute_metrics(y_true, y_scores, threshold)

def evaluate_hrv(model, X, y, threshold=None, target_sens=0.90):
    y = np.asarray(y)
    scores = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, scores)

    if threshold is None:
        valid = np.where(tpr >= target_sens)[0]
        if len(valid) == 0:
            threshold = 0.5
        else:
            specificity = 1 - fpr[valid]
            threshold = thresholds[valid[np.argmax(specificity)]]

    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auroc": auc(fpr, tpr),
        "auprc": average_precision_score(y, scores),
    }

def threshold_at_sensitivity(y_true, y_scores, target_sens=0.90):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    valid = np.where(tpr >= target_sens)[0]

    if len(valid) == 0:
        idx = np.argmax(tpr)
    else:
        min_fpr = np.min(fpr[valid])
        candidates = valid[np.isclose(fpr[valid], min_fpr)]
        idx = candidates[np.argmax(thresholds[candidates])]

    return thresholds[idx]


def threshold_at_max_fp_day(y_true, y_scores, max_fp_day=1.0, window_hours=1.0):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fp_day = fpr * (24.0 / window_hours)
    valid = np.where(fp_day <= max_fp_day)[0]

    if len(valid) == 0:
        return thresholds[np.argmin(fp_day)]

    max_sens = np.max(tpr[valid])
    candidates = valid[np.isclose(tpr[valid], max_sens)]

    return thresholds[candidates[np.argmax(thresholds[candidates])]]


def clinical_metrics(y_true, y_scores, threshold, window_hours=1.0):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    ppv = tp / (tp + fp) if tp + fp > 0 else np.nan
    npv = tn / (tn + fn) if tn + fn > 0 else np.nan
    f1 = (
        2 * ppv * sensitivity / (ppv + sensitivity)
        if ppv + sensitivity > 0 else np.nan
    )

    n_nsr = np.sum(y_true == 0)
    nsr_days = n_nsr * window_hours / 24.0
    fp_day = fp / nsr_days if nsr_days > 0 else np.nan

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "fp_day": fp_day,
        "roc_auc": auc(fpr, tpr),
        "auprc": average_precision_score(y_true, y_scores),
    }

evaluate_model_subj = evaluate_model_subject
