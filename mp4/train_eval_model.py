"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=50,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    N,D = data['image'].shape
    x_y = np.zeros((N,D+1))
    x_y[:,:-1] = data['image'].copy()
    x_y[:,-1:] = data['label'].copy()

    while num_steps > 0:
        if shuffle:
            np.random.shuffle(x_y)

        end = 0
        for start in range(0,N,batch_size):
            end = min(start + batch_size, N)
            update_step(x_y[start:end,:-1],x_y[start:end,-1:],model,learning_rate)
            num_steps -= 1
            if num_steps <= 0:
                break

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)
    g = model.backward(f,y_batch)

    model.w -= learning_rate * g


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    D = data['image'].shape[1]
    # Set model.w
    model.w = z[:D+1,:]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None

    # Implementation here
    N,D = data['image'].shape
    x = np.ones((N,D+1))
    x[:,:D] = data['image']
    y = data['label']

    P = np.zeros((N+D+1,N+D+1))
    P[:D+1,:D+1] = model.w_decay_factor * np.eye(D+1)

    q = np.ones((N+D+1,1))
    q[:D+1] = np.zeros((D+1,1))

    G = np.zeros((2*N,N+D+1))
    G[:N,:D+1] = y * x
    G[:N,D+1:] = np.eye(N)
    G[N:,D+1:] = np.eye(N)
    G = -1 * G

    h = -1 * np.ones((2*N,1))
    h[N:,:] = np.zeros((N,1))

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    x = data['image']
    y = data['label']
    f = model.forward(x)
    loss = model.total_loss(f,y)
    p = model.predict(f)
    acc = len(p[p==y])/len(y)

    return loss, acc
