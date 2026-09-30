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
