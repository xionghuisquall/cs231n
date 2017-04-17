import numpy as np
from random import shuffle

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

  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    # print 'it ', i, ': scores=', scores

    tmax = np.max(scores)
    # print tmax
    scores = scores - tmax
    # print 'it ', i, ': scores=', scores
    sum_e = np.sum(np.exp(scores))

    loss += -scores[y[i]] + np.log(sum_e)

    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += X[i] * (np.exp(scores[j]) / sum_e - 1)
      else:
        dW[:, j] += X[i] * (np.exp(scores[j]) / sum_e)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  score = X.dot(W)
  score_max = np.max(score, axis=1).reshape(-1, 1)
  score -= score_max

  score_correct = score[np.arange(num_train), y]
  score_exp = np.exp(score)
  score_exp_sum = np.sum(score_exp, axis=1)

  loss = -1 * score_correct + np.log(score_exp_sum)
  loss = np.mean(loss)

  # compute dW
  score_exp = score_exp / score_exp_sum.reshape(-1, 1)
  score_exp[np.arange(num_train), y] -= 1
  dW = X.transpose().dot(score_exp)
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW