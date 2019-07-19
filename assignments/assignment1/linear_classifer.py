import numpy as np
from math import exp, log

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    predictions_ = predictions - np.max(predictions);
    probs = np.array(list(map(exp, predictions_)));
    probs = probs/sum(probs);
    return probs;
    raise Exception("Not implemented!")


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
    
    raise Exception("Not implemented!")

    

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
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

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    
    # TODO: implement l2 regularization and gradient
    loss = reg_strength*sum(sum(W**2));
    grad = reg_strength*2*W;
    
    return loss, grad
    #raise Exception("Not implemented!")


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W);
    loss, dZ = softmax_with_cross_entropy(predictions, target_index);
    dW = np.dot(X.T, dZ);
    
    # TODO implement prediction and gradient over W
    return loss, dW
    #raise Exception("Not implemented!")
    

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            #print("shuffled_indeces:", shuffled_indices);
            
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            #print("bathes_indeces:", batches_indices);
            #print(num_train, num_features, num_classes);
            
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch in batches_indices:
                #print("y[batch]:", y[batch]);
                lin = linear_softmax(X[batch,:], self.W, y[batch]+1);
                l2 = l2_regularization(self.W, reg);
                loss, grad = lin[0] + l2[0], lin[1] + l2[1]
                loss_history.append(loss);
                self.W -= learning_rate*grad;
            if abs(loss_history[-1] - loss_history[-len(sections)]) < 1e-5:
                    print("Interrupt cycle of epochs(on epoch %i) for reason small difference losses between fit batches: %f" % (epoch, loss));
                    return loss_history;
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss), end = '\r');
        print("End epohs, loss_last: %f" % loss);
        return loss_history
        #raise Exception("Not implemented!")

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        predictions = np.dot(X, self.W)
        
        i=0;
        for predict in predictions:
            values = [softmax_with_cross_entropy(predict, target_index + 1)[0] \
                        for target_index in range(self.W.shape[1])];
            y_pred[i] = min(range(len(values)), key=values.__getitem__);
            i += 1;
        
        return y_pred


                
                                                          

            

                
