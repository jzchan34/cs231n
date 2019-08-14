import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W) #(1, C)
    correct_class = y[i]
    scores -= np.max(scores)
    denom = 0
    for c in range(num_classes):
      denom += np.exp(scores[c])
    p = np.exp(scores)/denom
    f = -np.log(p)
    loss += f[correct_class]
    for c in range(num_classes):
      if c == correct_class:
        x = p[c] - 1
      else:
        x = p[c] - 0
      dW[:, c] += x * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW  /= num_train
  loss += 0.5 *reg * np.sum(W * W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) #(N, C)
  #numeric instability fix
  max_vec = np.amax(scores, axis=1) #(N)
  scores = (scores.T - max_vec).T #(N,C)
  #loss
  denom = np.sum(np.exp(scores), axis=1) #(N)
  softmax = (np.exp(scores).T/denom).T #(N,C)
  loss_mat = -np.log(softmax)
  loss += np.sum(loss_mat[range(loss_mat.shape[0]), y])
  loss /= num_train
  loss += reg * np.sum(W * W)
  grad = softmax #(N,C)
  grad[range(grad.shape[0]), y] -= 1 
  dW = (X.T.dot(grad)) #(D,N) dot (N,C) = (D,C)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

