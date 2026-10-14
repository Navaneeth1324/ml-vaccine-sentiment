"""
util.py - Evaluation metrics, dataset loaders, confusion matrix, and reporting utilities.
Designed for UE24CS352A Machine Learning Mini-Project.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, and F1 Score for binary sentiment classification (0: Negative, 1: Positive).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total)
    }


def format_metrics_table(results: List[Dict[str, Any]], title: str = "Model Performance Comparison") -> str:
    """
    Formats evaluation results into a clean markdown table matching the paper's Table 1/2.
    """
    header = f"### {title}\n\n"
    header += "| Classifier | Tokenization | Dev Acc | Dev Precision | Dev Recall | Dev F1 |\n"
    header += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    
    rows = []
    for res in results:
        clf = res.get("classifier", "Unknown")
        tok = res.get("tokenization", "Standard")
        acc = f"{res.get('accuracy', 0.0):.4f}"
        prec = f"{res.get('precision', 0.0):.4f}"
        rec = f"{res.get('recall', 0.0):.4f}"
        f1 = f"{res.get('f1', 0.0):.4f}"
        rows.append(f"| {clf} | {tok} | {acc} | {prec} | {rec} | {f1} |")
        
    return header + "\n".join(rows) + "\n"


def load_dataset_csv(filepath: str) -> pd.DataFrame:
    """
    Loads dataset containing 'text', 'label', and optional 'month' / 'timestamp'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    return df


def save_dataset_csv(df: pd.DataFrame, filepath: str):
    """Saves DataFrame to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=False)


def format_confusion_matrix_ascii(metrics: Dict[str, Any], title: str = "Confusion Matrix") -> str:
    """
    Renders an ASCII confusion matrix representation from metric counts.
    """
    tp = metrics.get("tp", 0)
    tn = metrics.get("tn", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    
    matrix_str = f"\n=== {title} ===\n"
    matrix_str += f"                Predicted 0 (Neg)  Predicted 1 (Pos)\n"
    matrix_str += f"Actual 0 (Neg):        {tn:<18} {fp:<18}\n"
    matrix_str += f"Actual 1 (Pos):        {fn:<18} {tp:<18}\n"
    matrix_str += f"Total Samples: {tp + tn + fp + fn}\n"
    return matrix_str


def stratified_kfold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "label",
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into stratified train-validation folds preserving class ratios.
    Returns list of (train_fold_df, val_fold_df) pairs.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    rng = np.random.RandomState(random_state)
    folds = [[] for _ in range(n_splits)]
    
    for label_val in df[label_col].unique():
        sub_indices = df[df[label_col] == label_val].index.to_numpy()
        if shuffle:
            rng.shuffle(sub_indices)
        split_arrays = np.array_split(sub_indices, n_splits)
        for i in range(n_splits):
            folds[i].extend(split_arrays[i])
            
    result = []
    all_indices = set(df.index)
    for i in range(n_splits):
        val_idx = list(folds[i])
        train_idx = list(all_indices - set(val_idx))
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        result.append((train_df, val_df))
        
    return result


def classification_report_summary(y_true: List[int], y_pred: List[int]) -> str:
    """
    Generates a textual classification performance breakdown with per-class and overall metrics.
    """
    m = compute_metrics(y_true, y_pred)
    # Per-class negative (class 0)
    # TN is TP for negative, FP is FN for negative, etc.
    neg_prec = m["tn"] / (m["tn"] + m["fn"]) if (m["tn"] + m["fn"]) > 0 else 0.0
    neg_rec = m["tn"] / (m["tn"] + m["fp"]) if (m["tn"] + m["fp"]) > 0 else 0.0
    neg_f1 = (2 * neg_prec * neg_rec) / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0

    report = (
        f"Classification Report Summary\n"
        f"-----------------------------------------------------------\n"
        f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}\n"
        f"-----------------------------------------------------------\n"
        f"{'0 (Negative)':<15} {neg_prec:<12.4f} {neg_rec:<12.4f} {neg_f1:<12.4f} {m['tn'] + m['fp']:<8}\n"
        f"{'1 (Positive)':<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['tp'] + m['fn']:<8}\n"
        f"-----------------------------------------------------------\n"
        f"{'Accuracy':<15} {'':<12} {'':<12} {m['accuracy']:<12.4f} {m['total']:<8}\n"
    )
    return report

