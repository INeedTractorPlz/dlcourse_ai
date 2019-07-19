import numpy as np
from math import exp, log

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*sum(sum(W**2));
    grad = reg_strength*2*W;
    
    return loss, grad

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    #print("probs:", probs);
    
    return -log(probs[target_index - 1]);


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
   # TODO implement softmax with cross-entropy
    
    #One-dimension option
    
    if predictions.ndim == 1:
        predictions_ = predictions - np.max(predictions);
        dprediction = np.array(list(map(exp, predictions_)));
        summ = sum(dprediction);
        dprediction /= summ;
        
        loss = cross_entropy_loss(dprediction, target_index);
        dprediction[target_index - 1] -= 1;
        
        return loss, dprediction;
    else:
    
        predictions_ = predictions - np.max(predictions, axis = 1)[:, np.newaxis];
        exp_vec = np.vectorize(exp);
        #print("predictions_:", predictions_);
        
        dprediction = np.apply_along_axis(exp_vec, 1, predictions_);
        #print("dprediction before division: ", dprediction);
    
        summ = sum(dprediction.T);
        #print("summ: ", summ);
        dprediction /= summ[:, np.newaxis];
            
        #print("dprediction after division: ", dprediction);
    
        loss = np.array([cross_entropy_loss(x,y) for x,y in zip(dprediction, target_index)]);
        #print("loss: ", loss);
        
        #print("target_index - 1:", target_index - 1);
        it = np.nditer(target_index - 1, flags = ['c_index'] )
        while not it.finished:
            #print("it[0] = ", it[0]);
            dprediction[it.index, it[0]] -= 1
            it.iternext()
        
        dprediction /= len(target_index);
        #print("dprediction after subtraction: ", dprediction);
    
        return loss.mean(), dprediction;
    raise Exception("Not implemented!")


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
    #self.init = value.copy();
        self.value = value;
        self.grad = np.zeros_like(value);

class ReLULayer:
    def __init__(self):
        self.X = None
        
    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X;
        return (X > 0)*X;
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        return (self.X > 0)*d_out;

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.01 * np.random.randn(n_input, n_output))
        self.B = Param(0.01 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X;
        #if np.any(self.W.init != self.W.value) or np.any(self.B.init != self.B.value):
        self.W.grad = np.zeros_like(self.W.value);
        self.B.grad = np.zeros_like(self.B.value);
        #    self.W.init = self.W.value;
        #    self.B.init = self.B.value;
        return np.dot(self.X, self.W.value) + self.B.value;

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        dW = np.dot(self.X.T, d_out);
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out);
        
        d_input = np.dot(d_out, self.W.value.T);
        
        self.W.grad += dW;
        self.B.grad += dB;
        
        return d_input;

    def params(self):
        return {'W': self.W, 'B': self.B}
