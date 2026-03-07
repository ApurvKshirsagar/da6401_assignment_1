"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork
from ann.objective_functions import get_loss
from utils.data_loader import load_data

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d',   '--dataset',       type=str,   default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-mp',  '--model_path',    type=str,   default='best_model.npy')
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


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data
    
def evaluate_model(model, X_test, y_test,loss_fn, batch_size=64): 
    """
    Evaluate model on test data.
    """
    all_logits = []

    # run inference in batches to avoid memory issues
    for start in range(0, X_test.shape[0], batch_size):
        X_batch = X_test[start : start + batch_size]
        logits  = model.forward(X_batch)
        all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    preds  = logits.argmax(axis=1)

    # compute loss
    loss = loss_fn.forward(logits, y_test)

    # compute metrics
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    f1   = f1_score(y_test, preds, average='macro', zero_division=0)

    return {
        "logits"   : logits,
        "loss"     : loss,
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    # load test data only — no need for train/val
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    #Rebuild model architecture and load saved weights
    model   = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    loss_fn = get_loss(args.loss)
    results = evaluate_model(model, X_test, y_test, loss_fn, args.batch_size)

    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-score  : {results['f1']:.4f}")

    print("Evaluation complete!")
    return results
    
if __name__ == '__main__':
    main()
