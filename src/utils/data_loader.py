"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
#Libraries
import numpy as np

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]
MNIST_LABELS = [str(i) for i in range(10)]


def load_dataset(name, val_split=0.1, seed=42):
    name = name.lower().strip()

    if name == "mnist":
        from keras.datasets import mnist
        (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
        labels = MNIST_LABELS
    elif name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
        labels = FASHION_LABELS
    else:
        raise ValueError(f"Unknown dataset '{name}'. Use 'mnist' or 'fashion_mnist'.")

    # flatten + normalise
    X_tr = X_tr.reshape(-1, 784).astype(np.float64) / 255.0
    X_te = X_te.reshape(-1, 784).astype(np.float64) / 255.0
    y_tr = y_tr.astype(np.int64)
    y_te = y_te.astype(np.int64)

    # train / val split
    rng      = np.random.default_rng(seed)
    n_val    = int(len(X_tr) * val_split)
    shuffled = rng.permutation(len(X_tr))

    X_val,   y_val   = X_tr[shuffled[:n_val]],  y_tr[shuffled[:n_val]]
    X_train, y_train = X_tr[shuffled[n_val:]],  y_tr[shuffled[n_val:]]

    print(f"[Data] {name}  train={len(X_train)}  val={len(X_val)}  test={len(X_te)}")
    return X_train, y_train, X_val, y_val, X_te, y_te, labels