"""
test_error_analysis.py - Unit tests for qualitative error analysis and linguistic cue detection.
Author: Navaneeth Tanuboddi (PES1UG24CS709)
"""

import unittest
from src.error_analysis import (
    detect_linguistic_features,
    categorize_error,
    ErrorAnalyzer
)


class TestErrorAnalysis(unittest.TestCase):
    def test_detect_linguistic_features_negation(self):
        feat = detect_linguistic_features("I do not want to get vaccinated, no way!")
        self.assertTrue(feat["has_negation"])
        self.assertFalse(feat["has_contrast"])

        feat_zh = detect_linguistic_features("打完第一针没有任何副作用")
        self.assertTrue(feat_zh["has_negation"])

    def test_detect_linguistic_features_contrast(self):
        feat = detect_linguistic_features("Although I had mild fever, the vaccine was worth it.")
        self.assertTrue(feat["has_contrast"])

        feat_zh = detect_linguistic_features("虽然有点头痛，但是很快恢复了")
        self.assertTrue(feat_zh["has_contrast"])

    def test_detect_rhetorical(self):
        feat = detect_linguistic_features("难道这疫苗真的安全吗？")
        self.assertTrue(feat["has_rhetorical"])

    def test_categorize_error(self):
        cat1 = categorize_error("虽然有点难受，但是完全不影响", true_label=1, pred_label=0)
        self.assertEqual(cat1, "Mixed Affect + Negation")

        cat2 = categorize_error("疫苗不安全", true_label=0, pred_label=1)
        self.assertEqual(cat2, "Negation Reversal")

    def test_error_analyzer_initialization_and_fit(self):
        train_records = [
            {"text": "接种疫苗非常安全有效，支持！", "label": 1},
            {"text": "打完第一针没有任何不良反应，感谢医护人员", "label": 1},
            {"text": "疫苗出现严重不良反应，发烧难受", "label": 0},
            {"text": "听到好多副作用的新闻，太害怕了不敢打", "label": 0}
        ]
        dev_records = [
            {"id": "d1", "text": "接种疫苗很顺利，感觉很好", "label": 1},
            {"id": "d2", "text": "发烧难受，副作用好严重，完全不敢再打了", "label": 0}
        ]

        analyzer = ErrorAnalyzer()
        analyzer.fit_models(train_records)
        errors = analyzer.analyze(dev_records)
        self.assertIsInstance(errors, list)
        summary = analyzer.summarize_errors(errors, total_samples=len(dev_records))
        self.assertIn("overall_error_rate", summary)
        self.assertIn("categories", summary)


if __name__ == "__main__":
    unittest.main()
