"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import json
import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer
from utils.data_loader import load_data, create_batches
from sklearn.metrics import f1_score

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d',   '--dataset',       type=str,   default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e',   '--epochs',         type=int,   default=10)
    parser.add_argument('-b',   '--batch_size',     type=int,   default=64)
    parser.add_argument('-l',   '--loss',           type=str,   default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-o',   '--optimizer',      type=str,   default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr',  '--learning_rate',  type=float, default=0.001)
    parser.add_argument('-wd',  '--weight_decay',   type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers',     type=int,   default=3)
    parser.add_argument('-sz',  '--hidden_size',    type=int,   nargs='+', default=[128, 128, 128])
    parser.add_argument('-a',   '--activation',     type=str,   default='relu',
                        choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init',    type=str,   default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-w_p', '--wandb_project',  type=str,   default='da6401_a1')
    
    return parser.parse_args()

def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    # setup
    model     = NeuralNetwork(args)
    loss_fn   = get_loss(args.loss)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)

    best_f1      = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        steps      = 0

        for X_batch, y_batch in create_batches(X_train, y_train, args.batch_size):
            # forward
            logits = model.forward(X_batch)
            loss   = loss_fn.forward(logits, y_batch)

            # backward
            grad = loss_fn.backward()
            model.backward(grad)

            # update
            model.update_weights(optimizer)

            epoch_loss += loss
            steps      += 1

        # validation
        val_logits = model.forward(X_val)
        val_acc    = (val_logits.argmax(axis=1) == y_val).mean()

        # test f1 — used to pick best model as per assignment spec
        test_logits = model.forward(X_test)
        test_preds  = test_logits.argmax(axis=1)
        test_f1     = f1_score(y_test, test_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch:03d} | loss {epoch_loss/steps:.4f} | val_acc {val_acc:.4f} | test_f1 {test_f1:.4f}")

        # save best model by test f1
        if test_f1 > best_f1:
            best_f1      = test_f1
            best_weights = model.get_weights()

    # restore and save best weights
    model.set_weights(best_weights)
    np.save("best_model.npy", best_weights)
    print(f"\nBest test F1: {best_f1:.4f} — saved to best_model.npy")

    # save best config
    best_config = {
        "dataset"       : args.dataset,
        "epochs"        : args.epochs,
        "batch_size"    : args.batch_size,
        "loss"          : args.loss,
        "optimizer"     : args.optimizer,
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
        "num_layers"    : args.num_layers,
        "hidden_size"   : args.hidden_size,
        "activation"    : args.activation,
        "weight_init"   : args.weight_init
    }
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print("Config saved to best_config.json")

    print("Training complete!")


if __name__ == '__main__':
    main()