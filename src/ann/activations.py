"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
#Libraries
import numpy as np

class ReLU:
    def __init__(self):
        self.X = None 

    def forward(self,X):  
        self.X = X
        # f(X) = max(0,X)
        return np.maximum(0,X)

    def backward(self,dA):
        # f'(X) = 1 if X>0
        # f'(X) = 0 otherwise
        mask = (self.X > 0).astype(float)
        return dA * mask

class Sigmoid:
    def __init__(self):
        self.val = None 
    
    def forward(self,X):
        X = np.clip(X,-500,500) # prevent overflow
        # f(X) = 1 / (1 + e^(-X))
        self.val = 1.0 /(1.0 + np.exp(-X))
        return self.val 

    def backward(self,dA):
        # f'(X) = f(X)*(1-f(X))
        deriv = self.val * (1-self.val)
        return dA * deriv

class Tanh:
    def __init__(self):
        self.val = None 

    def forward(self,X):
        # f(X) = tanh(X) = (e^(X)-e^(-X))/(e^(X)+e^(-X)))
        self.val = np.tanh(X)
        return self.val

    def backward(self,dA):
        # f'(X) = 1 - f(X)^2
        deriv = 1 - self.val**2
        return dA*deriv

class Softmax:
    def __init__(self):
        self.val = None

    def forward(self,X):
        #Since exponentails can get pretty large we shift for numerical stability
        X_shifted= X - np.max(X,axis=1, keepdims=True)
        expo_vals = np.exp(X_shifted)

        self.val = expo_vals / np.sum(expo_vals,axis=1,keepdims=True)
        return self.val

    def backward(self, dA):
        #Softmax derivative uses Jacobian matrix ..directly defined in loss functions so did not write here
        pass

class Identity:
    def forward(self, X):
        self.X = X
        return X

    def backward(self, dA):
        return dA  #Gradient passes through unchanged

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "softmax":
        return Softmax()
    elif name in ("none", "identity"):
        return Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")