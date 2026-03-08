"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from ann.activations import Softmax


class CrossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        self.y_true  = None
        self.probs   = None

    def forward(self, X, y_true):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_true = np.atleast_1d(np.array(y_true))
        self.probs  = self.softmax.forward(X)
        self.y_true = y_true
        batch   = y_true.shape[0]
        correct = self.probs[np.arange(batch), y_true]
        return -np.log(correct + 1e-9).mean()

    def backward(self, y_true=None, logits=None):
        # supports both: backward() and backward(y_true, logits)
        if y_true is not None and logits is not None:
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            y_true = np.atleast_1d(np.array(y_true))
            self.forward(logits, y_true)
        batch = self.y_true.shape[0]
        dX    = self.probs.copy()
        dX[np.arange(batch), self.y_true] -= 1.0
        dX   /= batch
        return dX


class MSELoss:
    def __init__(self):
        self.softmax = Softmax()
        self.probs   = None
        self.diff    = None
        self.y_true  = None

    def forward(self, X, y_true):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_true = np.atleast_1d(np.array(y_true))
        self.y_true  = y_true
        self.probs   = self.softmax.forward(X)
        batch, n_classes = self.probs.shape
        one_hot      = np.zeros((batch, n_classes))
        one_hot[np.arange(batch), y_true] = 1.0
        self.diff    = self.probs - one_hot
        return (self.diff ** 2).mean()

    def backward(self, y_true=None, logits=None):
        # supports both: backward() and backward(y_true, logits)
        if y_true is not None and logits is not None:
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            self.forward(logits, y_true)
        batch, n_classes = self.diff.shape
        scale = 2.0 / (batch * n_classes)
        d     = self.diff * scale
        p     = self.probs
        dot   = (d * p).sum(axis=1, keepdims=True)
        return p * (d - dot)


def get_loss(name):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("cross_entropy", "ce"):
        return CrossEntropyLoss()
    elif name in ("mse", "mean_squared_error"):
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")