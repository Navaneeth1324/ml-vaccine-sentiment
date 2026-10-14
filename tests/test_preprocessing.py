"""
test_preprocessing.py - Unit tests for data cleaning, Unicode parsing, tokenization, and Vocabulary.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import unittest
from src.preprocessing import (
    clean_text,
    is_cjk_char,
    tokenize_unigram,
    tokenize_bigram,
    tokenize_trigram,
    tokenize,
    Vocabulary
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_bracket_tags(self):
        raw = "打了疫苗感觉很好[doge][泪]，加油！"
        cleaned = clean_text(raw, remove_bracket_tags=True)
        self.assertNotIn("[doge]", cleaned)
        self.assertNotIn("[泪]", cleaned)
        self.assertIn("打了疫苗感觉很好", cleaned)

    def test_clean_text_urls_and_mentions(self):
        raw = "Check this link https://example.com/vaccine and @doctor #COVID#"
        cleaned = clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("@doctor", cleaned)
        self.assertNotIn("#COVID#", cleaned)
        self.assertIn("Check this link", cleaned)

    def test_clean_text_whitespace_and_empty(self):
        self.assertEqual(clean_text("   multiple    spaces   "), "multiple spaces")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_is_cjk_char(self):
        self.assertTrue(is_cjk_char(ord("中")))
        self.assertTrue(is_cjk_char(ord("苗")))
        self.assertFalse(is_cjk_char(ord("A")))
        self.assertFalse(is_cjk_char(ord("1")))
        self.assertFalse(is_cjk_char(ord("!")))

    def test_tokenize_unigram(self):
        text = "COVID vaccine safe"
        tokens = tokenize_unigram(text, remove_stopwords=False)
        self.assertIn("covid", tokens)
        self.assertIn("vaccine", tokens)
        self.assertIn("safe", tokens)

    def test_tokenize_bigram(self):
        text = "接种 疫苗 安全"
        bigrams = tokenize_bigram(text, remove_stopwords=False)
        self.assertTrue(len(bigrams) >= 1)
        for bi in bigrams:
            self.assertIn("_", bi)

    def test_tokenize_trigram(self):
        text = "接种 新冠 疫苗"
        trigrams = tokenize_trigram(text, remove_stopwords=False)
        self.assertTrue(len(trigrams) >= 1)

    def test_tokenize_dispatcher(self):
        text = "Testing tokenization pipeline"
        uni = tokenize(text, method="unigram")
        bi = tokenize(text, method="bigram")
        tri = tokenize(text, method="trigram")
        self.assertIsInstance(uni, list)
        self.assertIsInstance(bi, list)
        self.assertIsInstance(tri, list)
        with self.assertRaises(ValueError):
            tokenize(text, method="unsupported_method")

    def test_vocabulary_encoding_decoding(self):
        corpus = [
            ["vaccine", "safe", "effective"],
            ["vaccine", "dose", "side", "effect"]
        ]
        vocab = Vocabulary(min_freq=1, max_vocab_size=50)
        vocab.build_vocab(corpus)

        self.assertIn("vaccine", vocab.token2id)
        encoded = vocab.encode(["vaccine", "safe", "unknown_word_xyz"])
        self.assertEqual(len(encoded), 3)
        # Check unknown token mapped to <UNK> (id 1)
        self.assertEqual(encoded[2], 1)

        decoded = vocab.decode(encoded)
        self.assertEqual(decoded[0], "vaccine")
        self.assertEqual(decoded[1], "safe")
        self.assertEqual(decoded[2], "<UNK>")


if __name__ == "__main__":
    unittest.main()
