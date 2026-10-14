"""
test_fact_checker.py - Unit tests for Grounded Misinformation Flagging & Fact-Check Verification Engine.
Author: Krrish P Raju (PES1UG24CS708)
"""

import unittest
from src.fact_checker import FactCheckVerifier, VERIFIED_FACT_DATABASE, FactCheckResult


class TestFactChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = FactCheckVerifier()

    def test_database_entries(self):
        self.assertGreater(len(VERIFIED_FACT_DATABASE), 0)
        for entry in VERIFIED_FACT_DATABASE:
            self.assertTrue(entry.topic)
            self.assertTrue(entry.grounded_fact)
            self.assertTrue(entry.source)
            self.assertGreater(len(entry.sample_claims), 0)

    def test_dna_alteration_misinformation(self):
        text = "mRNA vaccines alter human DNA and change your genes forever!"
        result = self.verifier.verify(text, sentiment_pred=0, sentiment_conf=0.94)

        self.assertEqual(result.sentiment_label, "Negative")
        self.assertAlmostEqual(result.sentiment_conf, 0.94)
        self.assertEqual(result.risk_level, "Elevated")
        self.assertIn("mRNA vaccines do not alter human DNA", result.grounded_fact)
        self.assertIn("CDC", result.grounded_fact)

        formatted = result.format_output()
        self.assertIn("Sentiment: Negative (0.94)", formatted)
        self.assertIn("Flag: Misinformation Risk: Elevated", formatted)
        self.assertIn('Grounded Fact: "CDC finding: mRNA vaccines do not alter human DNA.', formatted)

    def test_microchip_misinformation(self):
        text = "COVID vaccines contain 5G microchips to track and control people!"
        result = self.verifier.verify(text, sentiment_pred=0, sentiment_conf=0.91)

        self.assertEqual(result.risk_level, "Elevated")
        self.assertIn("microchips", result.grounded_fact.lower())

    def test_infertility_misinformation(self):
        text = "COVID-19 vaccines cause female infertility and sterilization."
        result = self.verifier.verify(text, sentiment_pred=0, sentiment_conf=0.88)

        self.assertEqual(result.risk_level, "Elevated")
        self.assertIn("infertility", result.grounded_fact.lower())

    def test_chinese_dna_misinformation(self):
        text = "听说打新冠疫苗会改变人体DNA和基因，千万别打！"
        result = self.verifier.verify(text, sentiment_pred=0, sentiment_conf=0.95)

        self.assertEqual(result.risk_level, "Elevated")
        self.assertIn("mRNA vaccines do not alter human DNA", result.grounded_fact)

    def test_benign_negative_post_not_falsely_flagged(self):
        # A normal personal report of a sore arm should not trigger a high myth flag
        text = "打完疫苗手臂酸痛了两天，有点难受。"
        result = self.verifier.verify(text, sentiment_pred=0, sentiment_conf=0.75)

        self.assertEqual(result.sentiment_label, "Negative")
        self.assertIn(result.risk_level, ["Low", "Moderate"])
        if result.risk_level == "Low":
            self.assertEqual(result.grounded_fact, "")

    def test_positive_post_has_low_risk(self):
        text = "Just got my second COVID vaccine dose, feeling protected and safe!"
        result = self.verifier.verify(text, sentiment_pred=1, sentiment_conf=0.98)

        self.assertEqual(result.sentiment_label, "Positive")
        self.assertEqual(result.risk_level, "Low")
        self.assertEqual(result.grounded_fact, "")


if __name__ == "__main__":
    unittest.main()
