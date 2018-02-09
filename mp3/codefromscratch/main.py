"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.0001
max_iters = 1000

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    X,Y_true = read_dataset('../data/trainset', 'indexing.txt')

    # Initialize model.
    model = LogisticModel(ndims=X.shape[1]-1, W_init='zeros')

    # Train model via gradient descent.
    model.fit(Y_true,X,learn_rate,max_iters)

    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')

    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')

    # Try all other methods: forward, backward, classify, compute accuracy
    result = model.classify(X) * Y_true
    tot = len(result)
    correct = len(result[result>0])
    print(correct/tot)    
