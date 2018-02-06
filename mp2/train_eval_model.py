"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent
    [x,y] = processed_dataset
    num_data = x.shape[0]

    count = 0
    stop = False
    while not stop:
        if shuffle:
            temp = np.zeros((x.shape[0],x.shape[1]+1))
            temp[:,:-1] = x
            temp[:,x.shape[1]:] = y
            np.random.shuffle(temp)
            x = temp[:,:-1]
            y = temp[:,-1]
            y = y.reshape(y.shape[0],1)

        for i in range(0, num_data, batch_size):
            idx = min(i + batch_size, num_data)
            update_step(x[i:idx], y[i:idx], model, learning_rate)
            count += 1
            if count == num_steps:
                stop = True
                break

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    grad = model.backward(f,y_batch)
    model.w -= learning_rate * grad


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    [x,y] = processed_dataset
    temp_x = np.ones((x.shape[0],x.shape[1]+1))
    temp_x[:,:-1] = x
    xTx = np.matmul(temp_x.T, temp_x)
    lambda_I = model.w_decay_factor*np.eye(xTx.shape[0])
    temp = xTx + lambda_I
    temp = np.linalg.inv(temp)
    temp = np.matmul(temp,temp_x.T)
    model.w = np.matmul(temp,y)


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    [x,y] = processed_dataset
    f = model.forward(x)
    loss = model.total_loss(f,y)

    return loss
