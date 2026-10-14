"""
test_models.py - Unit tests for all 5 machine learning and neural sequence models.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import unittest
import numpy as np
from src.naive_bayes import MultinomialNaiveBayes, InterpolatedNaiveBayes
from src.svm_classifier import SVMClassifier
from src.logistic_regression import LogisticRegressionClassifier
from src.bigram_model import BigramLanguageModelClassifier
from src.rnn_lstm_model import LSTMSentimentClassifier


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_texts = [
            "接种疫苗非常安全有效，支持！",
            "打完第一针没有任何副作用，感谢医护人员",
            "中国疫苗赞，相信科学相信国家",
            "疫苗出现严重不良反应，发烧难受",
            "接种后胳膊痛了好几天，很后悔去打",
            "听到好多副作用的新闻，太害怕了不敢打"
        ]
        cls.train_labels = [1, 1, 1, 0, 0, 0]
        cls.test_texts = [
            "接种疫苗很顺利，感觉很好",
            "发烧头晕难受，副作用严重"
        ]

    def test_multinomial_naive_bayes(self):
        clf = MultinomialNaiveBayes(tokenization="unigram")
        clf.fit(self.train_texts, self.train_labels)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], [0, 1])
        probs = clf.predict_proba(self.test_texts[0])
        self.assertIn(0, probs)
        self.assertIn(1, probs)
        self.assertAlmostEqual(probs[0] + probs[1], 1.0, places=4)

    def test_interpolated_naive_bayes(self):
        clf = InterpolatedNaiveBayes(lmbda=0.8)
        clf.fit(self.train_texts, self.train_labels)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)

    def test_svm_classifier(self):
        clf = SVMClassifier(tokenization="bigram", C=1.0)
        clf.fit(self.train_texts, self.train_labels)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], [0, 1])

    def test_logistic_regression(self):
        clf = LogisticRegressionClassifier(tokenization="bigram", C=1.0)
        clf.fit(self.train_texts, self.train_labels)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)
        probs = clf.predict_proba(self.test_texts)
        self.assertEqual(probs.shape, (2, 2))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-5)

    def test_bigram_language_model(self):
        clf = BigramLanguageModelClassifier(alpha=1.0)
        clf.fit(self.train_texts, self.train_labels)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)
        probs = clf.predict_proba(self.test_texts[0])
        self.assertAlmostEqual(probs[0] + probs[1], 1.0, places=4)

    def test_lstm_sentiment_classifier(self):
        clf = LSTMSentimentClassifier(vocab_size=100, embed_dim=16, hidden_dim=16, random_state=42)
        clf.fit(self.train_texts, self.train_labels, epochs=1)
        preds = clf.predict(self.test_texts)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0], [0, 1])
        probs = clf.predict_proba(self.test_texts[0])
        self.assertTrue(0.0 <= probs[1] <= 1.0)
        self.assertAlmostEqual(probs[0] + probs[1], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
