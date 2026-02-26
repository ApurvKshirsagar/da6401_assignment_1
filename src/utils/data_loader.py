"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name, validation_split=0.1, random_state=42):
    """
    Loads MNIST or Fashion-MNIST dataset.
    Performs:
        - normalization
        - flattening
        - one-hot encoding
        - train/validation split
    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    """

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")

    # Normalize to [0,1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten (N, 28, 28) → (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One-hot encode labels
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=random_state,
        shuffle=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def one_hot_encode(y, num_classes):
    """
    Converts labels (N,) to one-hot (N, num_classes)
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def create_batches(X, y, batch_size):
    """
    Generator that yields mini-batches.
    """

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        yield X[start_idx:end_idx], y[start_idx:end_idx]