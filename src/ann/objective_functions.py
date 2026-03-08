"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

#Libraries
import numpy as np
from ann.activations import Softmax

class CrossEntropyLoss: #Categorical Cross-Entropy
    def __init__(self):
        self.softmax = Softmax()
        self.y_true = None
        self.probs = None

    # def forward(self,X, y_true):
    #     #X is logits
    #     self.probs = self.softmax.forward(X)
    #     self.y_true = y_true
    #     batch = y_true.shape[0]
    #     correct = self.probs[np.arange(batch),y_true]
    #     return -np.log(correct + 1e-9).mean() #1e-9 is added to prevent log(0) from blowing up to -infinity

    def forward(self, X, y_true):
        # ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # ensure y_true is 1D array
        y_true = np.atleast_1d(np.array(y_true))
        
        self.probs = self.softmax.forward(X)
        self.y_true = y_true
        batch = y_true.shape[0]
        correct = self.probs[np.arange(batch), y_true]
        return -np.log(correct + 1e-9).mean()

    def backward(self): 
        #The actual derivate of softmax is complex, but when we use crossentropy with softmax, most things cancels and remaining term is simple
        # softmax + cross entropy gradient simplifies to (p - one_hot) / N
        batch = self.y_true.shape[0]
        dX = self.probs.copy()
        dX[np.arange(batch), self.y_true] -= 1.0
        dX /= batch
        return dX

class MSELoss:
    def __init__(self):
        self.softmax = Softmax()
        self.probs = None
        self.diff = None

    # def forward(self, X, y_true):
    #     self.probs = self.softmax.forward(X)
    #     batch, n_classes = X.shape
    #     # OneHot Matrix: 
    #     one_hot = np.zeros((batch, n_classes))
    #     one_hot[np.arange(batch), y_true] = 1.0
    #     #Difference between prediction and target
    #     self.diff = self.probs - one_hot
    #     return (self.diff ** 2).mean()

    def forward(self, X, y_true):
        # ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # ensure y_true is 1D array
        y_true = np.atleast_1d(np.array(y_true))

        self.probs = self.softmax.forward(X)
        batch, n_classes = self.probs.shape
        one_hot = np.zeros((batch, n_classes))
        one_hot[np.arange(batch), y_true] = 1.0
        self.diff = self.probs - one_hot
        return (self.diff ** 2).mean()

    def backward(self):
        batch, n_classes = self.diff.shape
        scale = 2.0 / (batch * n_classes)
        d = self.diff * scale
        p = self.probs
        dot = (d * p).sum(axis=1, keepdims=True) #coupling term
        return p * (d - dot)

def get_loss(name):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("cross_entropy", "ce"):
        return CrossEntropyLoss()
    elif name in ("mse", "mean_squared_error"):
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")