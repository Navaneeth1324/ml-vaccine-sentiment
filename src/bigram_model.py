"""
bigram_model.py - Bigram Markov Language Model Sentiment Classifier.
Directly implements Equation (6) from Section 4.4 of the reference paper:
p(y | x) proportional to p(e | x_n, y) * p(x_n | x_{n-1}, y) * ... * p(x_2 | x_1, y) * p(x_1 | b, y) * p(y)
with Add-1 (Laplace) smoothing.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from src.preprocessing import clean_text, tokenize_unigram


class BigramLanguageModelClassifier:
    """
    Bigram generative classifier modeling sequence transition probabilities per sentiment class.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Add-1 Laplace smoothing
        self.vocab = set()
        self.unigram_counts = {0: Counter(), 1: Counter()}
        self.bigram_counts = {0: defaultdict(Counter), 1: defaultdict(Counter)}
        self.class_priors = {0: 0.5, 1: 0.5}
        self.vocab_size = 0
        self.START_TOKEN = "<b>"
        self.END_TOKEN = "<e>"

    def fit(self, texts: List[str], labels: List[int]):
        """
        Fits class-conditional bigram transition distributions.
        """
        self.vocab = {self.START_TOKEN, self.END_TOKEN}
        self.unigram_counts = {0: Counter(), 1: Counter()}
        self.bigram_counts = {0: defaultdict(Counter), 1: defaultdict(Counter)}
        
        class_doc_counts = Counter(labels)
        total_docs = len(labels)
        self.class_priors[0] = (class_doc_counts[0] + 1) / (total_docs + 2)
        self.class_priors[1] = (class_doc_counts[1] + 1) / (total_docs + 2)

        for text, label in zip(texts, labels):
            cleaned = clean_text(text)
            tokens = tokenize_unigram(cleaned)
            # Add start and end tokens
            seq = [self.START_TOKEN] + tokens + [self.END_TOKEN]
            
            for tok in seq:
                self.vocab.add(tok)
                self.unigram_counts[label][tok] += 1
                
            for i in range(len(seq) - 1):
                w_prev = seq[i]
                w_curr = seq[i + 1]
                self.bigram_counts[label][w_prev][w_curr] += 1

        self.vocab_size = len(self.vocab)

    def _bigram_log_prob(self, w_prev: str, w_curr: str, label: int) -> float:
        """
        Smoothed bigram transition probability:
        p(w_curr | w_prev, y) = (count(w_prev, w_curr, y) + 1) / (count(w_prev, y) + |V|)
        """
        bi_count = self.bigram_counts[label][w_prev][w_curr]
        uni_count = self.unigram_counts[label][w_prev]
        prob = (bi_count + self.alpha) / (uni_count + self.alpha * self.vocab_size + 1e-9)
        return math.log(prob)

    def predict_log_score(self, text: str) -> Dict[int, float]:
        """
        Computes log p(y|x) = log p(y) + sum_{i=1}^{n+1} log p(x_i | x_{i-1}, y).
        """
        cleaned = clean_text(text)
        tokens = tokenize_unigram(cleaned)
        seq = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        scores = {}
        for c in (0, 1):
            score = math.log(self.class_priors[c])
            for i in range(len(seq) - 1):
                score += self._bigram_log_prob(seq[i], seq[i + 1], c)
            scores[c] = score
        return scores

    def predict_proba(self, text: str) -> Dict[int, float]:
        scores = self.predict_log_score(text)
        s0, s1 = scores[0], scores[1]
        max_s = max(s0, s1)
        exp0 = math.exp(s0 - max_s)
        exp1 = math.exp(s1 - max_s)
        tot = exp0 + exp1
        return {0: exp0 / tot, 1: exp1 / tot}

    def predict(self, texts: List[str]) -> List[int]:
        preds = []
        for text in texts:
            scores = self.predict_log_score(text)
            preds.append(1 if scores[1] >= scores[0] else 0)
        return preds
