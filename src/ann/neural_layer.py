"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

#Libraries
import numpy as np 
from ann.activations import get_activation

class DenseLayer:
    def __init__(self, in_dim, out_dim,activation="relu", w_init="random"):
        # Either xavier or random initialistion
        if(w_init == "xavier"):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        else:
            multi = 0.01  #To ensure small values
            self.W = np.random.randn(in_dim,out_dim) * multi

        self.b = np.zeros((1, out_dim))
        self.X = None
        self.grad_W = None
        self.grad_b = None
        self.activation = get_activation(activation) 

    # def forward(self, X):
    #     self.X = X
    #     Z =  np.dot(X, self.W) + self.b  #As Z = XW + b
    #     return self.activation.forward(Z) #A = g(Z)

    # def backward(self, dA):
    #     #Chain rule through activation first
    #     dZ = self.activation.backward(dA)

    #     m = self.X.shape[0] 
    #     self.grad_W = np.dot(self.X.T, dZ) / m
    #     self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

    #     # Pass gradient back to previous layer
    #     dX = np.dot(dZ, self.W.T)
    #     return dX

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.X = X
        Z = np.dot(X, self.W) + self.b
        return self.activation.forward(Z)

    def backward(self, dA):
        if dA.ndim == 1:
            dA = dA.reshape(1, -1)
        dZ = self.activation.backward(dA)
        m = self.X.shape[0]
        self.grad_W = np.dot(self.X.T, dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.W.T)
        return dX