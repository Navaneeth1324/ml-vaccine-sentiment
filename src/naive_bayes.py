"""
naive_bayes.py - Naive Bayes Classifier with Laplace smoothing and Bigram/Trigram linear interpolation.
Directly implements Equations (1), (2), and (3) from Section 4.1 of the reference research paper.
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
from src.preprocessing import clean_text, tokenize


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier with Laplace (add-1) smoothing.
    """
    def __init__(self, alpha: float = 1.0, tokenization: str = "unigram"):
        self.alpha = alpha  # Laplace smoothing constant (+1)
        self.tokenization = tokenization
        self.vocab = set()
        self.class_counts = Counter()  # Number of documents per class
        self.token_counts = {0: Counter(), 1: Counter()}  # Count of each word in class
        self.total_tokens = {0: 0, 1: 0}  # Total word tokens in class
        self.log_prior = {0: 0.0, 1: 0.0}
        self.vocab_size = 0

    def fit(self, texts: List[str], labels: List[int]):
        """
        Trains the Naive Bayes classifier on text documents and binary labels (0: Negative, 1: Positive).
        """
        self.vocab = set()
        self.class_counts = Counter()
        self.token_counts = {0: Counter(), 1: Counter()}
        self.total_tokens = {0: 0, 1: 0}
        
        n_samples = len(texts)
        for text, label in zip(texts, labels):
            cleaned = clean_text(text)
            tokens = tokenize(cleaned, method=self.tokenization)
            self.class_counts[label] += 1
            for tok in tokens:
                self.vocab.add(tok)
                self.token_counts[label][tok] += 1
                self.total_tokens[label] += 1

        self.vocab_size = len(self.vocab)
        
        # Calculate class log priors
        for c in (0, 1):
            count = self.class_counts[c]
            self.log_prior[c] = math.log((count + self.alpha) / (n_samples + 2 * self.alpha))

    def _get_log_likelihood(self, token: str, label: int) -> float:
        """
        Calculates Laplace smoothed conditional likelihood: P(w_k | y = label).
        Equation (1) & (2) from the paper.
        """
        count = self.token_counts[label].get(token, 0)
        # Smoothed probability: (count + alpha) / (total_tokens + alpha * |V|)
        prob = (count + self.alpha) / (self.total_tokens[label] + self.alpha * self.vocab_size + 1e-9)
        return math.log(prob)

    def predict_log_posterior(self, text: str) -> Dict[int, float]:
        """
        Computes log P(Y=y) + sum(log P(w|Y=y)).
        """
        cleaned = clean_text(text)
        tokens = tokenize(cleaned, method=self.tokenization)
        
        scores = {}
        for c in (0, 1):
            score = self.log_prior[c]
            for tok in tokens:
                score += self._get_log_likelihood(tok, c)
            scores[c] = score
        return scores

    def predict_proba(self, text: str) -> Dict[int, float]:
        """
        Returns normalized probability estimates P(y=0|x) and P(y=1|x) using softmax over log posteriors.
        """
        scores = self.predict_log_posterior(text)
        s0, s1 = scores[0], scores[1]
        max_s = max(s0, s1)
        exp0 = math.exp(s0 - max_s)
        exp1 = math.exp(s1 - max_s)
        total = exp0 + exp1
        return {0: exp0 / total, 1: exp1 / total}

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predicts classes for a list of input documents.
        """
        preds = []
        for text in texts:
            scores = self.predict_log_posterior(text)
            preds.append(1 if scores[1] > scores[0] else 0)
        return preds


class InterpolatedNaiveBayes:
    """
    Interpolated Naive Bayes combining Bigram and Trigram predictions.
    Direct implementation of Equation (3) from Section 4.1:
    log(P(Y=1|X)) = lambda * log(P_bigram(Y=1|X)) + (1 - lambda) * log(P_trigram(Y=1|X))
    """
    def __init__(self, lmbda: float = 0.8):
        self.lmbda = lmbda
        self.nb_bigram = MultinomialNaiveBayes(tokenization="bigram")
        self.nb_trigram = MultinomialNaiveBayes(tokenization="trigram")

    def fit(self, texts: List[str], labels: List[int]):
        self.nb_bigram.fit(texts, labels)
        self.nb_trigram.fit(texts, labels)

    def predict_proba(self, text: str) -> Dict[int, float]:
        prob_bi = self.nb_bigram.predict_proba(text)
        prob_tri = self.nb_trigram.predict_proba(text)
        
        # Combine probabilities weighted by lambda
        p1 = self.lmbda * prob_bi[1] + (1 - self.lmbda) * prob_tri[1]
        p0 = 1.0 - p1
        return {0: p0, 1: p1}

    def predict(self, texts: List[str]) -> List[int]:
        preds = []
        for text in texts:
            prob = self.predict_proba(text)
            preds.append(1 if prob[1] >= prob[0] else 0)
        return preds
