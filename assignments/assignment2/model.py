import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.first = FullyConnectedLayer(n_input, hidden_layer_size);
        self.ReLU = ReLULayer();
        self.second = FullyConnectedLayer(hidden_layer_size, n_output);
        self.n_output = n_output;
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params():
            self.params()[param].grad = np.zeros_like(self.params()[param].grad);
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        X1 = self.first.forward(X);
        X1_ReLU = self.ReLU.forward(X1);
        X2 = self.second.forward(X1_ReLU);
        
        loss, dX2 = softmax_with_cross_entropy(X2, y + 1);
        dX1_ReLU = self.second.backward(dX2);
        dX1 = self.ReLU.backward(dX1_ReLU);
        dX = self.first.backward(dX1);
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params():
            if(param[0] == 'W'):
                loss_, grad_ = l2_regularization(self.params()[param].value, self.reg);
                self.params()[param].grad += grad_;
                loss += loss_;
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        predictions = self.second.forward(self.ReLU.forward(self.first.forward(X)));
        i=0;
        for predict in predictions:
            values = [softmax_with_cross_entropy(predict, target_index + 1)[0] \
                        for target_index in range(self.n_output)];
            pred[i] = min(range(len(values)), key=values.__getitem__);
            i += 1;
        return pred

    def params(self):
        result = {}
        dict_first = self.first.params();
        dict_second = self.second.params();
        # TODO Implement aggregating all of the params
        
        for key in dict_first.keys():
            result[key + '1'] = dict_first[key];
        
        for key in dict_second.keys():
            result[key + '2'] = dict_second[key];
        
        return result
