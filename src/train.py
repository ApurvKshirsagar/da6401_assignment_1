"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import json
import argparse
import numpy as np
import wandb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer
from utils.data_loader import load_data, create_batches

matplotlib.use('Agg')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d',   '--dataset',       type=str,   default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e',   '--epochs',        type=int,   default=10)
    parser.add_argument('-b',   '--batch_size',    type=int,   default=64)
    parser.add_argument('-l',   '--loss',          type=str,   default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-o',   '--optimizer',     type=str,   default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr',  '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd',  '--weight_decay',  type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers',    type=int,   default=3)
    parser.add_argument('-sz',  '--hidden_size',   type=int,   nargs='+', default=[128, 128, 128])
    parser.add_argument('-a',   '--activation',    type=str,   default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init',   type=str,   default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-w_p', '--wandb_project', type=str,   default='da6401_a1')
    return parser.parse_args()


def log_sample_images(X_train, y_train, dataset_name):
    class_names_mnist   = ['0','1','2','3','4','5','6','7','8','9']
    class_names_fashion = ['T-shirt','Trouser','Pullover','Dress','Coat',
                           'Sandal','Shirt','Sneaker','Bag','Ankle boot']
    names = class_names_fashion if dataset_name == "fashion_mnist" else class_names_mnist
    table = wandb.Table(columns=["image", "label", "class_name"])
    for class_idx in range(10):
        indices = np.where(y_train == class_idx)[0][:5]
        for idx in indices:
            img = X_train[idx].reshape(28, 28)
            table.add_data(wandb.Image(img), class_idx, names[class_idx])
    wandb.log({"sample_images": table})


def get_dead_neuron_fraction(model, X_sample):
    dead_fractions = []
    x = X_sample
    for layer in model.layers[:-1]:
        z = x @ layer.W + layer.b
        a = layer.activation.forward(z)
        if layer.activation.__class__.__name__ == 'ReLU':
            dead = (a == 0).all(axis=0).mean()
        else:
            dead = 0.0
        dead_fractions.append(dead)
        x = a
    return dead_fractions


def log_confusion_matrix(model, X_test, y_test):
    test_logits = model.forward(X_test)
    test_preds  = test_logits.argmax(axis=1)
    test_probs  = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)

    # confusion matrix
    cm  = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im  = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Best Model on MNIST Test Set')
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

    # failure analysis table
    table       = wandb.Table(columns=["image", "true_label", "pred_label", "confidence"])
    wrong_idx   = np.where(test_preds != y_test)[0]
    confidences = test_probs[wrong_idx, test_preds[wrong_idx]]
    top_wrong   = wrong_idx[np.argsort(confidences)[::-1][:50]]
    for idx in top_wrong:
        img  = X_test[idx].reshape(28, 28)
        true = y_test[idx]
        pred = test_preds[idx]
        conf = test_probs[idx, pred]
        table.add_data(wandb.Image(img), int(true), int(pred), float(conf))
    wandb.log({"failure_analysis": table})


def main():
    args = parse_arguments()

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    wandb.init(project=args.wandb_project)

    config = wandb.config
    if hasattr(config, 'learning_rate'): args.learning_rate = config.learning_rate
    if hasattr(config, 'optimizer'):     args.optimizer     = config.optimizer
    if hasattr(config, 'activation'):    args.activation    = config.activation
    if hasattr(config, 'num_layers'):    args.num_layers    = config.num_layers
    if hasattr(config, 'hidden_size'):   args.hidden_size   = [config.hidden_size] * args.num_layers
    if hasattr(config, 'weight_decay'):  args.weight_decay  = config.weight_decay
    if hasattr(config, 'batch_size'):    args.batch_size    = config.batch_size
    if hasattr(config, 'epochs'):        args.epochs        = config.epochs

    wandb.config.update(vars(args), allow_val_change=True)

    if not wandb.run.sweep_id:
        log_sample_images(X_train, y_train, args.dataset)

    model     = NeuralNetwork(args)
    loss_fn   = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)

    best_f1      = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        steps      = 0

        for X_batch, y_batch in create_batches(X_train, y_train, args.batch_size):
            logits = model.forward(X_batch)
            loss   = loss_fn.forward(logits, y_batch)
            grad   = loss_fn.backward()
            model.backward(grad)
            model.update_weights(optimizer)
            epoch_loss += loss
            steps      += 1

        train_logits = model.forward(X_train)
        train_acc    = (train_logits.argmax(axis=1) == y_train).mean()

        sample_X       = X_train[:500]
        dead_fractions = get_dead_neuron_fraction(model, sample_X)
        dead_neuron_log = {}
        for i, frac in enumerate(dead_fractions):
            dead_neuron_log[f"dead_neurons_layer{i}"] = frac

        val_logits  = model.forward(X_val)
        val_acc     = (val_logits.argmax(axis=1) == y_val).mean()

        test_logits = model.forward(X_test)
        test_preds  = test_logits.argmax(axis=1)
        test_f1     = f1_score(y_test, test_preds, average='macro', zero_division=0)

        grad_norm = np.linalg.norm(model.layers[0].grad_W) if model.layers[0].grad_W is not None else 0.0

        print(f"Epoch {epoch:03d} | loss {epoch_loss/steps:.4f} | val_acc {val_acc:.4f} | test_f1 {test_f1:.4f}")
        wandb.log({
            "loss"            : epoch_loss / steps,
            "train_acc"       : train_acc,
            "val_acc"         : val_acc,
            "test_f1"         : test_f1,
            "grad_norm_layer0": grad_norm,
            **dead_neuron_log
        })

        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()

    model.set_weights(best_weights)
    np.save("best_model.npy", best_weights)
    print(f"\nBest test F1: {best_f1:.4f} — saved to best_model.npy")

    if not wandb.run.sweep_id:
        best_config = {
            "dataset"      : args.dataset,
            "epochs"       : args.epochs,
            "batch_size"   : args.batch_size,
            "loss"         : args.loss,
            "optimizer"    : args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay" : args.weight_decay,
            "num_layers"   : args.num_layers,
            "hidden_size"  : args.hidden_size,
            "activation"   : args.activation,
            "weight_init"  : args.weight_init
        }
        with open("best_config.json", "w") as f:
            json.dump(best_config, f, indent=4)
        print("Config saved to best_config.json")

        log_confusion_matrix(model, X_test, y_test)

    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()