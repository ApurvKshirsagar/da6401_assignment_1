"""
Hyperparameter Sweep for DA6401 Assignment 1
"""
import wandb

sweep_config = {
    "program": "train.py", 
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {"values": [0.001, 0.005, 0.01, 0.0001]},
        "optimizer"    : {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "activation"   : {"values": ["relu", "sigmoid", "tanh"]},
        "num_layers"   : {"values": [2, 3, 4]},
        "hidden_size"  : {"values": [64, 128]},
        "weight_decay" : {"values": [0.0, 0.0001, 0.001]},
        "batch_size"   : {"values": [32, 64, 128]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="da6401_a1")
print(f"Sweep created!")
print(f"Now run: wandb agent ce22b042-iit-madras/da6401_a1/{sweep_id} --count 100")