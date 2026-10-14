"""
test_trend_analysis.py - Unit tests for monthly sentiment aggregation and time series plotting.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import os
import unittest
import tempfile
import pandas as pd
from src.trend_analysis import compute_monthly_summary, plot_sentiment_vs_vaccine_doses


class DummyModel:
    def predict(self, texts):
        # Predict 1 for every text
        return [1] * len(texts)


class TestTrendAnalysis(unittest.TestCase):
    def test_compute_monthly_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "monthly_posts.csv")
            data = {
                "month": ["December 31", "December 31", "January", "January", "January"],
                "text": ["post 1", "post 2", "post 3", "post 4", "post 5"],
                "ground_truth_label": [1, 0, 1, 1, 0]
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)

            models = {"Dummy": DummyModel()}
            summary_df = compute_monthly_summary(csv_path, models)

            self.assertIn("Month", summary_df.columns)
            self.assertIn("Sample Count", summary_df.columns)
            self.assertIn("Dummy Pos %", summary_df.columns)
            self.assertEqual(len(summary_df), 2)  # Dec 31 and Jan

    def test_plot_sentiment_vs_vaccine_doses(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            timeline_csv = os.path.join(tmp_dir, "timeline.csv")
            output_img = os.path.join(tmp_dir, "test_fig.png")

            timeline_data = {
                "date": ["2020-12-31", "2021-01-31"],
                "month": ["Dec 31", "Jan 31"],
                "doses_millions": [4.5, 24.0],
                "sentiment_pct": [78.5, 72.0]
            }
            pd.DataFrame(timeline_data).to_csv(timeline_csv, index=False)

            plot_sentiment_vs_vaccine_doses(timeline_csv, output_img)
            self.assertTrue(os.path.exists(output_img))
            self.assertGreater(os.path.getsize(output_img), 0)


if __name__ == "__main__":
    unittest.main()
