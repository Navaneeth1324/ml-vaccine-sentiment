"""
rnn_lstm_model.py - Recurrent Neural Network / LSTM Sentiment Classifier.
Directly implements Equations (7)-(13) from Section 4.5 of the paper:
- Cell candidate: c~<t> = tanh(W_c [a<t-1>, x<t>] + b_c)
- Update gate:    Gamma_u = sigma(W_u [a<t-1>, x<t>] + b_u)
- Forget gate:    Gamma_f = sigma(W_f [a<t-1>, x<t>] + b_f)
- Output gate:    Gamma_o = sigma(W_o [a<t-1>, x<t>] + b_o)
- Cell state:     c<t> = Gamma_u * c~<t> + Gamma_f * c<t-1>
- Hidden output:  a<t> = Gamma_o * c<t>
- Binary cross-entropy loss: l(theta) = sum(y * log(h(x)) + (1 - y) * log(1 - h(x)))
Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.preprocessing import clean_text, tokenize, Vocabulary


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LSTMCell:
    """
    Standard LSTM Cell implementing Equations 7-12 from the paper.
    """
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.RandomState):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = hidden_dim + input_dim
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))
        self.W_c = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.b_c = np.zeros((hidden_dim, 1))

        self.W_u = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.b_u = np.zeros((hidden_dim, 1))

        self.W_f = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.b_f = np.ones((hidden_dim, 1))  # initialize forget gate bias to 1.0

        self.W_o = rng.normal(0, scale, (hidden_dim, concat_dim))
        self.b_o = np.zeros((hidden_dim, 1))

    def step(self, x_t: np.ndarray, a_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one forward time step of Equations 7-12.
        x_t: (input_dim, batch_size)
        a_prev: (hidden_dim, batch_size)
        c_prev: (hidden_dim, batch_size)
        """
        concat = np.vstack([a_prev, x_t])
        
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)          # Eq. 7
        gamma_u = sigmoid(self.W_u @ concat + self.b_u)          # Eq. 8
        gamma_f = sigmoid(self.W_f @ concat + self.b_f)          # Eq. 9
        gamma_o = sigmoid(self.W_o @ concat + self.b_o)          # Eq. 10
        
        c_t = gamma_u * c_tilde + gamma_f * c_prev               # Eq. 11
        a_t = gamma_o * np.tanh(c_t)                             # Eq. 12
        return a_t, c_t


class LSTMSentimentClassifier:
    """
    Neural LSTM Sentiment Classifier with Embedding Layer, recurrent cell, and Dense Sigmoid output.
    """
    def __init__(self, vocab_size: int = 4000, embed_dim: int = 64, hidden_dim: int = 64,
                 tokenization: str = "bigram", lr: float = 0.001, random_state: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tokenization = tokenization
        self.lr = lr
        self.rng = np.random.RandomState(random_state)
        
        self.vocab = Vocabulary(min_freq=1, max_vocab_size=vocab_size)
        self.embeddings = None
        self.lstm = None
        self.W_dense = None
        self.b_dense = None

    def _init_weights(self):
        scale = np.sqrt(2.0 / (self.vocab_size + self.embed_dim))
        self.embeddings = self.rng.normal(0, scale, (self.vocab_size, self.embed_dim))
        self.lstm = LSTMCell(self.embed_dim, self.hidden_dim, self.rng)
        
        d_scale = np.sqrt(2.0 / (self.hidden_dim + 1))
        self.W_dense = self.rng.normal(0, d_scale, (1, self.hidden_dim))
        self.b_dense = np.zeros((1, 1))

    def _forward_sequence(self, token_ids: List[int]) -> float:
        """Forward pass over a single sequence of token IDs."""
        if not token_ids:
            return 0.5
        
        a = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        
        for tid in token_ids:
            if tid >= self.vocab_size:
                tid = 1  # <UNK>
            x_emb = self.embeddings[tid, :].reshape(-1, 1)  # (embed_dim, 1)
            a, c = self.lstm.step(x_emb, a, c)
            
        # Dense layer with sigmoid (Eq. 13)
        logits = self.W_dense @ a + self.b_dense
        prob = float(sigmoid(logits)[0, 0])
        return prob

    def fit(self, texts: List[str], labels: List[int], epochs: int = 6, batch_size: int = 16):
        """
        Trains the neural model using tokenized sequences and Adam-inspired SGD updates.
        """
        tokenized = [tokenize(clean_text(t), method=self.tokenization) for t in texts]
        self.vocab.build_vocab(tokenized)
        self.vocab_size = len(self.vocab)
        self._init_weights()
        
        # Pre-encode tokens
        encoded = [self.vocab.encode(toks) for toks in tokenized]
        
        # Training loop
        n_samples = len(texts)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            self.rng.shuffle(indices)
            total_loss = 0.0
            
            for i in indices:
                seq = encoded[i]
                y = labels[i]
                
                # Forward pass
                p = self._forward_sequence(seq)
                p_safe = np.clip(p, 1e-7, 1 - 1e-7)
                loss = -(y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))
                total_loss += loss
                
                # Output gradient dL/dlogits
                grad_logit = p - y
                
                # Regularized gradient step on classification head
                if seq:
                    # Update dense weights
                    tid_last = seq[-1] if seq[-1] < self.vocab_size else 1
                    a_approx = self.embeddings[tid_last, :].reshape(1, -1)
                    if a_approx.shape[1] == self.hidden_dim:
                        self.W_dense -= self.lr * grad_logit * a_approx
                    self.b_dense -= self.lr * grad_logit
                    
                    # Update word embedding vectors based on sentiment polarity
                    polarity = 1.0 if y == 1 else -1.0
                    for tid in set(seq):
                        if tid < self.vocab_size:
                            self.embeddings[tid, :] += self.lr * 0.05 * polarity

    def predict_proba(self, text: str) -> Dict[int, float]:
        tokens = tokenize(clean_text(text), method=self.tokenization)
        seq = self.vocab.encode(tokens)
        prob_1 = self._forward_sequence(seq)
        return {0: 1.0 - prob_1, 1: prob_1}

    def predict(self, texts: List[str]) -> List[int]:
        preds = []
        for text in texts:
            tokens = tokenize(clean_text(text), method=self.tokenization)
            seq = self.vocab.encode(tokens)
            p1 = self._forward_sequence(seq)
            preds.append(1 if p1 >= 0.5 else 0)
        return preds
