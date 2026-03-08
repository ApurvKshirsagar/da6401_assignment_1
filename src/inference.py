"""
Inference Script
Evaluate trained models on test sets
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    p = argparse.ArgumentParser(
        description="MLP Inference — MNIST / Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # architecture args — must match saved model
    p.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",   "--epochs",        type=int,   default=30)
    p.add_argument("-b",   "--batch_size",    type=int,   default=128)
    p.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                   choices=["cross_entropy", "mse"])
    p.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd",  "--weight_decay",  type=float, default=0.0001)
    p.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    p.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    p.add_argument("-a",   "--activation",    type=str,   default="tanh",
                   choices=["relu", "sigmoid", "tanh"])
    p.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                   choices=["xavier", "random"])
    p.add_argument("-w_p", "--wandb_project", type=str,   default="da6401_a1")
    # inference-specific
    p.add_argument("--model_path",  type=str,   default="best_model.npy")
    p.add_argument("--config_path", type=str,   default="best_config.json")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no_wandb",    action="store_true", default=True)
    return p.parse_args()


def load_weights(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No weights found at '{path}'. Run train.py first.")
    return np.load(path, allow_pickle=True).item()


def apply_config(args, config_path):
    if not os.path.exists(config_path):
        print(f"[Config] '{config_path}' not found — using CLI args.")
        return args
    with open(config_path) as f:
        cfg = json.load(f)
    for key in ["dataset", "num_layers", "hidden_size", "activation",
                "weight_init", "loss", "optimizer", "learning_rate",
                "weight_decay", "batch_size"]:
        if key in cfg:
            setattr(args, key, cfg[key])
    print(f"[Config] Loaded from '{config_path}'.")
    return args


def run_inference(model, X_test, y_test, loss_name):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
    from ann.objective_functions import get_loss

    # batched forward pass
    chunk    = 512
    logits   = np.concatenate(
        [model.forward(X_test[i: i + chunk]) for i in range(0, len(X_test), chunk)],
        axis=0
    )
    preds    = logits.argmax(axis=1)
    loss_val = get_loss(loss_name).forward(y_test, logits)

    return {
        "logits":           logits,
        "predictions":      preds,
        "loss":             float(loss_val),
        "accuracy":         float(accuracy_score(y_test, preds)),
        "precision":        float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall":           float(recall_score(y_test, preds,    average="macro", zero_division=0)),
        "f1":               float(f1_score(y_test, preds,         average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }


def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    args = apply_config(args, args.config_path)

    _, _, _, _, X_test, y_test, label_names = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    model = NeuralNetwork(args)
    model.set_weights(load_weights(args.model_path))
    print(f"[Inference] Weights loaded from '{args.model_path}'.")

    results = run_inference(model, X_test, y_test, args.loss)

    print("\n── Inference Results ──")
    print(f"  Dataset   : {args.dataset}  ({X_test.shape[0]} samples)")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1        : {results['f1']:.4f}")

    from sklearn.metrics import classification_report
    print("\nPer-class breakdown:")
    print(classification_report(y_test, results["predictions"],
                                target_names=label_names, zero_division=0))
    return results


if __name__ == "__main__":
    main()