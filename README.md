# DA6401 Assignment 1 — Multi-Layer Perceptron from Scratch

## By Apurv Ravindra Kshirsagar - CE22B042

Implementation of a configurable MLP using only NumPy for MNIST and Fashion-MNIST classification.

## W&B Report

[View the full experiment report here](https://wandb.ai/ce22b042-iit-madras/da6401_a1/reports/DA6401-Assignment-1--VmlldzoxNjEzNDYxNA?accessToken=vv3ta8qgzu2e6rng61ckh1ambdf0zqf88zyvzg15ikg85nvp73gj71jeclaqe0jw)

## Github Link

https://github.com/ApurvKshirsagar/da6401_assignment_1

## Project Structure

```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── train.py
│   ├── inference.py
│   └── sweep.py
```

## Setup Instructions

### 1. Create Conda Environment (Mac M2 / Apple Silicon)

```bash
CONDA_SUBDIR=osx-arm64 conda create -n da6401 python=3.10
conda activate da6401
conda config --env --set subdir osx-arm64
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib scikit-learn wandb
pip install tensorflow-macos tensorflow-metal
```

### 3. Login to W&B

```bash
wandb login
```

### 4. Navigate to src directory

```bash
cd da6401_assignment_1/src
```

## Training

### Basic Training

```bash
python train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -nhl 3 -sz 128 128 128 -a relu -w_i xavier -w_p da6401_a1
```

### All CLI Arguments

| Argument      | Flag | Default       | Description                 |
| ------------- | ---- | ------------- | --------------------------- |
| dataset       | -d   | mnist         | mnist or fashion_mnist      |
| epochs        | -e   | 30            | Number of training epochs   |
| batch_size    | -b   | 128           | Batch size                  |
| loss          | -l   | cross_entropy | cross_entropy or mse        |
| optimizer     | -o   | rmsprop       | sgd, momentum, nag, rmsprop |
| learning_rate | -lr  | 0.001         | Learning rate               |
| weight_decay  | -wd  | 0.0001        | L2 regularization           |
| num_layers    | -nhl | 3             | Number of hidden layers     |
| hidden_size   | -sz  | 128 128 128   | Neurons per hidden layer    |
| activation    | -a   | tanh          | sigmoid, tanh, relu         |
| weight_init   | -w_i | xavier        | random, xavier, zeros       |
| wandb_project | -w_p | da6401_a1     | W&B project name            |

## Hyperparameter Sweep

```bash
python sweep.py
wandb agent ce22b042-iit-madras/da6401_a1/SWEEP_ID --count 100
```

## Inference

```bash
python inference.py --model_path best_model.npy --config_path best_config.json
```

## Best Model Configuration

```json
{
  "dataset": "mnist",
  "epochs": 50,
  "batch_size": 128,
  "loss": "cross_entropy",
  "optimizer": "rmsprop",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "num_layers": 3,
  "hidden_size": [128, 128, 128],
  "activation": "tanh",
  "weight_init": "xavier"
}
```

To reproduce the best model:

```bash
python train.py -d mnist -e 50 -b 128 -o rmsprop -lr 0.001 -wd 0.0001 -nhl 3 -sz 128 128 128 -a tanh -w_i xavier -w_p da6401_a1
```

## What to Expect

- MNIST test accuracy: ~97-98%
- Fashion-MNIST test accuracy: ~88-90%
- Training time: ~2 minutes per run on Mac M2
- Sweep (100 runs): ~2-3 hours

## Dependencies

- numpy
- matplotlib
- scikit-learn
- wandb
- tensorflow-macos (for data loading only)
- tensorflow-metal (for Apple Silicon GPU)
