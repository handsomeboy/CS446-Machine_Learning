"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.0001
max_iters = 1000

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset','indexing.txt')
    X, Y_true = read_dataset_tf('../data/trainset', 'indexing.txt')
    
    # Initialize model.
    model = LogisticModel_TF(X.shape[1]-1, 'zeros')

    # Build TensorFlow training graph
    model.build_graph(learn_rate)  

    # Train model via gradient descent.
    score = model.fit(Y_true, X, max_iters)

    # Compute classification accuracy based on the return of the "fit" method
    accuracy = len(Y_true[np.round(score)==Y_true])/Y_true.shape[0]
    print(accuracy)

    
if __name__ == '__main__':
    tf.app.run()
