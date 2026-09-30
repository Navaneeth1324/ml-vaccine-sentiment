"""
logistic_regression.py - Regularized Logistic Regression Classifier.
Implements the regularized logistic loss formulation from Section 4.3 (Equation 5) of the paper:
min 1/2 * ||theta||^2 + C * sum(log(exp(-y_i * (theta^T * x_i + c)) + 1))
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import clean_text, tokenize


class LogisticRegressionClassifier:
    """
    Logistic Regression sentiment classifier with L2 regularization and flexible tokenization.
    """
    def __init__(self, tokenization: str = "bigram", C: float = 1.0, max_iter: int = 1000):
        self.tokenization = tokenization
        self.C = C
        self.max_iter = max_iter

        def custom_tokenizer(doc: str) -> List[str]:
            return tokenize(clean_text(doc), method=self.tokenization)

        self.vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,
            ngram_range=(1, 1),
            token_pattern=None,
            min_df=1,
            sublinear_tf=True
        )
        self.model = SklearnLogReg(C=self.C, max_iter=self.max_iter, solver="lbfgs", random_state=42)

    def fit(self, texts: List[str], labels: List[int]):
        """Trains Logistic Regression model."""
        X_vec = self.vectorizer.fit_transform(texts)
        self.model.fit(X_vec, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """Predicts binary sentiment class (0 or 1)."""
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Computes calibrated probability distributions [P(y=0|x), P(y=1|x)]."""
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_vec)
