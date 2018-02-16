"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1, 1).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        reg_grad = self.w_decay_factor * self.w

        one_yf = 1-y*f
        one_yf[one_yf > 0] = 1
        one_yf[one_yf <=0] = 0
        loss_grad = -1 * y * one_yf * self.x
        loss_grad = np.sum(loss_grad, axis=0)
        loss_grad = loss_grad.reshape(loss_grad.shape[0],1)

        total_grad = reg_grad + loss_grad
        #print(np.linalg.norm(total_grad))

        return total_grad


    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        hinge_loss = None
        l2_loss = None
        # Implementation here.
        z = np.zeros((y.shape[0],1))
        hinge_loss = np.sum(np.maximum(z,1-f*y))
        l2_loss = np.square(self.w).sum() * self.w_decay_factor / 2

        total_loss = hinge_loss + l2_loss

        return total_loss


    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        y_predict = None
        # Implementation here.
        y_predict = f.copy()
        y_predict[y_predict < 0] = -1
        y_predict[y_predict >=0] = 1

        return y_predict
