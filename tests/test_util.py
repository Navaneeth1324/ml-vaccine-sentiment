"""
test_util.py - Unit tests for evaluation metrics, reporting, and dataset utilities.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from src.util import (
    compute_metrics,
    format_metrics_table,
    format_confusion_matrix_ascii,
    stratified_kfold_split,
    classification_report_summary,
    load_dataset_csv,
    save_dataset_csv
)


class TestUtilMetrics(unittest.TestCase):
    def test_compute_metrics_perfect_prediction(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["tp"], 3)
        self.assertEqual(m["tn"], 2)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)

    def test_compute_metrics_partial_accuracy(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        # TP: 1 (idx 0), TN: 1 (idx 3), FP: 1 (idx 1), FN: 1 (idx 2)
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_compute_metrics_zero_division_guard(self):
        # Case where no positives are predicted
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["accuracy"], 1.0)

    def test_format_metrics_table(self):
        results = [
            {"classifier": "SVM", "tokenization": "Unigram", "accuracy": 0.85, "precision": 0.84, "recall": 0.86, "f1": 0.85}
        ]
        table = format_metrics_table(results, "Test Table")
        self.assertIn("### Test Table", table)
        self.assertIn("| SVM | Unigram | 0.8500 | 0.8400 | 0.8600 | 0.8500 |", table)

    def test_format_confusion_matrix_ascii(self):
        metrics = {"tp": 10, "tn": 15, "fp": 2, "fn": 3}
        matrix_str = format_confusion_matrix_ascii(metrics, "Model Evaluation")
        self.assertIn("=== Model Evaluation ===", matrix_str)
        self.assertIn("Total Samples: 30", matrix_str)

    def test_classification_report_summary(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]
        report = classification_report_summary(y_true, y_pred)
        self.assertIn("Classification Report Summary", report)
        self.assertIn("0 (Negative)", report)
        self.assertIn("1 (Positive)", report)

    def test_stratified_kfold_split(self):
        df = pd.DataFrame({
            "text": [f"sample_{i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50
        })
        folds = stratified_kfold_split(df, n_splits=5, shuffle=True, random_state=42)
        self.assertEqual(len(folds), 5)
        
        for train_fold, val_fold in folds:
            self.assertEqual(len(train_fold), 80)
            self.assertEqual(len(val_fold), 20)
            # Check 50:50 class balance preservation
            val_pos = (val_fold["label"] == 1).sum()
            val_neg = (val_fold["label"] == 0).sum()
            self.assertEqual(val_pos, 10)
            self.assertEqual(val_neg, 10)

    def test_csv_save_and_load(self):
        df = pd.DataFrame({"text": ["test post 1", "test post 2"], "label": [1, 0]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "test.csv")
            save_dataset_csv(df, tmp_file)
            self.assertTrue(os.path.exists(tmp_file))
            loaded_df = load_dataset_csv(tmp_file)
            self.assertEqual(len(loaded_df), 2)
            self.assertListEqual(loaded_df["label"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
