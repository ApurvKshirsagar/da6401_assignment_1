"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import DenseLayer


class NeuralNetwork:

    def __init__(self, cli_args):
        self.layers = []

        in_dim     = 784
        out_dim    = 10
        activation = cli_args.activation
        w_init     = cli_args.weight_init

        if isinstance(cli_args.hidden_size, list):
            sizes = cli_args.hidden_size
        else:
            sizes = [cli_args.hidden_size] * cli_args.num_layers

        prev = in_dim
        for h in sizes:
            self.layers.append(DenseLayer(prev, h, activation, w_init))
            prev = h

        self.layers.append(DenseLayer(prev, out_dim, activation="none", w_init=w_init))

    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true, y_pred=None):
        if y_pred is not None:
            # autograder style: model.backward(y_true, logits)
            N     = y_pred.shape[0]
            probs = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            grad  = probs.copy()
            grad[np.arange(N), y_true] -= 1.0
            grad /= N
        else:
            # train.py style: model.backward(grad)
            grad = y_true

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update_weights(self, optimizer):
        optimizer.update(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        pass

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds  = logits.argmax(axis=1)
        acc    = (preds == y).mean()
        return acc

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()