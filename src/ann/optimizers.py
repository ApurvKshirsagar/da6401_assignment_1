"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            grad_W = layer.grad_W + self.weight_decay * layer.W #L2 regularisation
            #Move in the opp side of the gradient
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {} #Stores velocities for each layer

    def update(self, layers):
        for idx, layer in enumerate(layers):
             # initialise velocity as zeros on first call
            if idx not in self.velocities:
                self.velocities[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            v = self.velocities[idx]
            v["vW"] = self.beta * v["vW"] + (1 - self.beta) * layer.grad_W
            v["vb"] = self.beta * v["vb"] + (1 - self.beta) * layer.grad_b

            layer.W -= self.lr * (v["vW"] + self.weight_decay * layer.W)
            layer.b -= self.lr * v["vb"]


class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if idx not in self.velocities:
                self.velocities[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            v = self.velocities[idx]

            vW_prev = v["vW"].copy()
            vb_prev = v["vb"].copy()

            v["vW"] = self.beta * v["vW"] + self.lr * (layer.grad_W + self.weight_decay * layer.W)
            v["vb"] = self.beta * v["vb"] + self.lr * layer.grad_b

            # nesterov correction: overshoot then pull back
            layer.W -= (1 + self.beta) * v["vW"] - self.beta * vW_prev
            layer.b -= (1 + self.beta) * v["vb"] - self.beta * vb_prev


class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon  #Prevents division by zero
        self.weight_decay = weight_decay
        self.cache = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if idx not in self.cache:
                self.cache[idx] = {
                    "sW": np.zeros_like(layer.W),
                    "sb": np.zeros_like(layer.b)
                }

            s = self.cache[idx]
            grad_W = layer.grad_W + self.weight_decay * layer.W

            s["sW"] = self.beta * s["sW"] + (1 - self.beta) * grad_W ** 2
            s["sb"] = self.beta * s["sb"] + (1 - self.beta) * layer.grad_b ** 2

            layer.W -= self.lr * grad_W / (np.sqrt(s["sW"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(s["sb"]) + self.eps)

def get_optimizer(name, learning_rate, weight_decay=0.0):
    opts = {
        "sgd":      SGD(learning_rate, weight_decay),
        "momentum": Momentum(learning_rate, weight_decay=weight_decay),
        "nag":      NAG(learning_rate, weight_decay=weight_decay),
        "rmsprop":  RMSProp(learning_rate, weight_decay=weight_decay),
    }
    if name not in opts:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {list(opts.keys())}")
    return opts[name]