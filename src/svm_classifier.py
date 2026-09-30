"""
svm_classifier.py - Support Vector Machine Classifier with N-gram / TF-IDF representations.
Implements the primal optimization problem from Section 4.2 of the paper:
min 1/2 * ||theta||^2 + C * sum(zeta_i) subject to y_i * (theta^T * phi(x_i) + b) >= 1 - zeta_i
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from src.preprocessing import clean_text, tokenize


class SVMClassifier:
    """
    Support Vector Machine text classifier supporting unigram, bigram, trigram, and TF-IDF features.
    """
    def __init__(self, tokenization: str = "bigram", C: float = 1.0, kernel: str = "linear"):
        self.tokenization = tokenization
        self.C = C
        self.kernel = kernel
        
        # Tokenizer hook for TfidfVectorizer
        def custom_tokenizer(doc: str) -> List[str]:
            return tokenize(clean_text(doc), method=self.tokenization)

        self.vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,
            ngram_range=(1, 1),
            token_pattern=None,
            min_df=1,
            sublinear_tf=True
        )
        
        if kernel == "linear":
            self.model = LinearSVC(C=self.C, max_iter=2000, random_state=42)
        else:
            self.model = SVC(C=self.C, kernel=self.kernel, probability=True, random_state=42)

    def fit(self, texts: List[str], labels: List[int]):
        """Trains the SVM on texts and binary labels."""
        X_vec = self.vectorizer.fit_transform(texts)
        self.model.fit(X_vec, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """Predicts binary classes."""
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Estimates probability or decision boundary distance."""
        X_vec = self.vectorizer.transform(texts)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_vec)
        else:
            decision = self.model.decision_function(X_vec)
            # Sigmoid calibration over decision scores
            probs = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - probs, probs]).T
