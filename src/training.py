import numpy as np

# from tinygrad.jit import TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.state import get_parameters
from tinygrad.tensor import Tensor
from extra.datasets import fetch_mnist

from .model import MnistModel, train_model
from .optimizer import sparse_cross_entropy

def run_training_mnist(use_new_backward:bool = False):
    # Sets variables
    n_epochs = 1000
    n_valid  = 10000

    # Retrieve dataset
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # Normalize
    X_train /= 255.
    X_test  /= 255.

    # Split training and validation sets
    indexes_valid = np.random.choice(range(len(X_train)), 
                                     n_valid, 
                                     replace=False)
    mask = np.ones(Y_train.shape, bool)
    mask[indexes_valid] = False

    X_val, Y_val = X_train[indexes_valid], Y_train[indexes_valid]
    X_train, Y_train = X_train[mask], Y_train[mask]

    # Generate network
    network    = MnistModel()
    net_params = get_parameters(network)
    optimizer  = SGD(net_params, lr=0.0005)

    # Train model
    network = train_model(
        network, 
        sparse_cross_entropy, 
        optimizer,
        X_train, 
        Y_train, 
        X_val, 
        Y_val,
        32,
        n_epochs,
        use_new_backward
    )
