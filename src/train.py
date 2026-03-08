"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
#Libraries
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
        description="MLP Trainer — MNIST / Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d",    "--dataset",       type=str,   default="mnist",
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("-e",    "--epochs",        type=int,   default=30)
    p.add_argument("-b",    "--batch_size",    type=int,   default=128)
    p.add_argument("-l",    "--loss",          type=str,   default="cross_entropy",
                   choices=["cross_entropy", "mse"])
    p.add_argument("-o",    "--optimizer",     type=str,   default="rmsprop",
                   choices=["sgd", "momentum", "nag", "rmsprop"])
    p.add_argument("-lr",   "--learning_rate", type=float, default=0.001)
    p.add_argument("-wd",   "--weight_decay",  type=float, default=0.0001)
    p.add_argument("-nhl",  "--num_layers",    type=int,   default=3)
    p.add_argument("-sz",   "--hidden_size",   type=int,   nargs="+", default=[128, 128, 128])
    p.add_argument("-a",    "--activation",    type=str,   default="tanh",
                   choices=["relu", "sigmoid", "tanh"])
    p.add_argument("-w_i", "--weight_init", type=str, default="xavier",
               choices=["xavier", "random", "zeros"])
    p.add_argument("-w_p",  "--wandb_project", type=str,   default="da6401_a1")
    p.add_argument("--wandb_entity",           type=str,   default=None)
    p.add_argument("--no_wandb",               action="store_true")
    p.add_argument("--model_path",             type=str,   default="best_model.npy")
    p.add_argument("--config_path",            type=str,   default="best_config.json")
    p.add_argument("--val_split",              type=float, default=0.1)
    p.add_argument("--seed",                   type=int,   default=42)
    return p.parse_args()


def setup_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        print(f"[W&B] {run.url}")
        return run
    except Exception as err:
        print(f"[W&B] Skipping — {err}")
        return None


def compute_metrics(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    result = model.evaluate(X, y)
    preds  = result["predictions"]
    return {
        "accuracy":    float(accuracy_score(y, preds)),
        "precision":   float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall":      float(recall_score(y, preds,    average="macro", zero_division=0)),
        "f1":          float(f1_score(y, preds,         average="macro", zero_division=0)),
        "loss":        result["loss"],
        "predictions": preds,
        "logits":      result["logits"],
    }


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    model = NeuralNetwork(args)
    arch  = " → ".join(str(s) for s in (
        args.hidden_size if isinstance(args.hidden_size, list) else [args.hidden_size]
    ))
    print(f"\nArchitecture : 784 → {arch} → 10")
    print(f"Optimizer    : {args.optimizer}  lr={args.learning_rate}  wd={args.weight_decay}")
    print(f"Loss         : {args.loss}   Activation: {args.activation}   Init: {args.weight_init}")
    print(f"Batch size   : {args.batch_size}   Epochs: {args.epochs}\n")

    wandb_run = setup_wandb(args)

    model.train(X_train, y_train, X_val=X_val, y_val=y_val, wandb_run=wandb_run)

    print("\n── Test Set Results ──")
    metrics = compute_metrics(model, X_test, y_test)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Loss      : {metrics['loss']:.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "test_accuracy":  metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall":    metrics["recall"],
            "test_f1":        metrics["f1"],
            "test_loss":      metrics["loss"],
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or ".", exist_ok=True)
    np.save(args.model_path, model.get_weights())
    print(f"\n[Saved] weights → {args.model_path}")

    config_out = {**vars(args), "test_f1": metrics["f1"], "test_accuracy": metrics["accuracy"]}
    with open(args.config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"[Saved] config  → {args.config_path}")

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDone!")
    return metrics


if __name__ == "__main__":
    main()