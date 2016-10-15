"""
MIT License

Copyright (c) 2016 Pythonix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pickle
import time
import math


class FeedForwardNetwork():

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=False, dropout_prop=0.5):
        self.input_layer = np.array([])
        self.hidden_layer = np.array([])
        self.output_layer = np.array([])
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.dropout_prop = dropout_prop
                       
        self.weights_input_hidden = np.random.uniform(low=-0.01, high=0.01, size=(input_dim, hidden_dim))
        self.weights_hidden_output = np.random.uniform(low=-0.01, high=0.01, size=(hidden_dim, output_dim))
        
        self.validation_data = np.array([])
        self.validation_data_solution = np.array([])
        
        self.velocities_input_hidden = np.zeros(self.weights_input_hidden.shape)
        self.velocities_hidden_output = np.zeros(self.weights_hidden_output.shape)

    def _tanh(self, x, deriv=False):
        #The derivate is: 1-np.tanh(x)**2; Because x is already the output of tanh(x) 1-x*x is the correct derivate.
        if not deriv:
            return np.tanh(x)
        return 1-x*x

    def _softmax(self, x, deriv=False):
        if not deriv:
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        return 1 - np.exp(x) / np.sum(np.exp(x), axis=0)

    def set_training_data(self, training_data_input, training_data_target, validation_data_input=None, validation_data_target=None):
        """Splits the data up into training and validation data with a ratio of 0.85/0.15 if no validation data is given.
        Sets the data for training."""
        if len(training_data_input) != len(training_data_target):
            raise ValueError(
                'Number of training examples and'
                ' training targets does not match!'
            )
        if (validation_data_input is None) and (validation_data_target is None):
            len_training_data = int((len(training_data_input)/100*85//1))
            self.input_layer = training_data_input[:len_training_data]
            self.output_layer = training_data_target[:len_training_data]
            self.validation_data = training_data_input[len_training_data:]
            self.validation_data_solution = training_data_target[len_training_data:]
        else:
            self.input_layer = training_data_input
            self.output_layer = training_data_target
            self.validation_data = validation_data_input
            self.validation_data_solution = validation_data_target

    def save(self, filename):
        """Saves the weights into a pickle file."""
        with open(filename, "wb") as network_file:
            pickle.dump(self.weights_input_hidden, network_file)
            pickle.dump(self.weights_hidden_output, network_file)

    def load(self, filename):
        """Loads network weights from a pickle file."""
        with open(filename, "rb") as network_file:
            weights_input_hidden = pickle.load(network_file)
            weights_hidden_output = pickle.load(network_file)
            
        if (
            len(weights_input_hidden) != len(self.weights_input_hidden)
            or len(weights_hidden_output) != len(self.weights_hidden_output)
        ):
            raise ValueError(
                'File contains weights that does not'
                ' match the current networks size!'
            )        
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output

    def measure_error(self, input_data, output_data):
        return 1/2 * np.sum((output_data - self.forward_propagate(input_data))**2)
        #return np.sum(np.nan_to_num(-output_data*np.log(self.forward_propagate(input_data))-(1-output_data)*np.log(1-self.forward_propagate(input_data))))

    def forward_propagate(self, input_data, dropout=False):
        """Proceds the input data from input neurons up to output neurons and returns the output layer.
           If dropout is True some of the neurons are randomly turned off."""
        input_layer = input_data
        self.hidden_layer = self._tanh(np.dot(input_layer, self.weights_input_hidden))
        if dropout:
            self.hidden_layer *= np.random.binomial([np.ones((len(input_data),self.hidden_dim))],1-self.dropout_prop)[0] * (1.0/(1-self.dropout_prop))
        return self._softmax(np.dot(self.hidden_layer, self.weights_hidden_output).T).T
        #return self._softmax(output_layer.T).T

    def back_propagate(self, input_data, output_data, alpha, beta, momentum):
        """Calculates the difference between target output and output and adjusts the weights to fit the target output better.
           The parameter alpha is the learning rate.
           Beta is the parameter for weight decay which penaltizes large weights."""
        sample_count = len(input_data)
        output_layer = self.forward_propagate(input_data, dropout=self.dropout)
        output_layer_error = output_layer - output_data
        output_layer_delta = output_layer_error * self._softmax(output_layer, deriv=True)
        #How much did each hidden neuron contribute to the output error?
        #Multiplys delta term with weights
        hidden_layer_error = output_layer_delta.dot(self.weights_hidden_output.T)
        
        #If the prediction is good, the second term will be small and the change will be small
        #Ex: target: 1 -> Slope will be 1 so the second term will be big
        hidden_layer_delta = hidden_layer_error * self._tanh(self.hidden_layer, deriv=True)
        #The both lines return a matrix. A row stands for all weights connected to one neuron.
        #E.g. [1, 2, 3] -> Weights to Neuron A
        #     [4, 5, 6] -> Weights to Neuron B
        hidden_weights_gradient = input_data.T.dot(hidden_layer_delta)/sample_count
        output_weights_gradient = self.hidden_layer.T.dot(output_layer_delta)/sample_count
        velocities_input_hidden = self.velocities_input_hidden
        velocities_hidden_output = self.velocities_hidden_output

        self.velocities_input_hidden = velocities_input_hidden * momentum - alpha * hidden_weights_gradient
        self.velocities_hidden_output = velocities_hidden_output * momentum - alpha * output_weights_gradient
                
        #Includes momentum term and weight decay; The weight decay parameter is beta
        #Weight decay penalizes large weights to prevent overfitting
        self.weights_input_hidden += -velocities_input_hidden * momentum + (1 + momentum) * self.velocities_input_hidden
        - alpha * beta * self.weights_input_hidden / sample_count
        self.weights_hidden_output += -velocities_hidden_output * momentum + (1 + momentum) * self.velocities_hidden_output
        - alpha * beta * self.weights_hidden_output / sample_count
        
    def batch_train(self, epochs, alpha, beta, momentum, patience=10, auto_save=False, show=50):
        """Trains the network in batch mode that means the weights are updated after showing all training examples.
           alpha is the learning rate and patience is the number of epochs that the validation error is allowed to increase before aborting.
           Beta is the parameter for weight decay which penaltizes large weights."""
        patience_const = patience
        validation_error = self.measure_error(self.validation_data, self.validation_data_solution)
        for epoch in range(epochs):
            self.back_propagate(self.input_layer, self.output_layer, alpha, beta, momentum)
            validation_error_new = self.measure_error(self.validation_data, self.validation_data_solution)
            if  validation_error_new < validation_error:
                validation_error = validation_error_new
                patience = patience_const
                if auto_save:
                    self.save(auto_save)
            else:
                patience -= 1
                if patience == 0:
                    print("Abort Training. Overfitting has started! Epoch: {0}. Error: {1}".format(epoch, validation_error_new))
                    return False
            if epoch % show == 0:
                print("Epoch: {0}, Validation Error: {1}".format(epoch, validation_error))
        return True

    def mini_batch_train(self, batch_size, epochs, alpha, beta, momentum, patience=10, auto_save=False, show=50):
        """Trains the network in mini batch mode, that means the weights are updated after showing only a bunch of training examples.
           alpha is the learning rate and patience is the number of epochs that the validation error is allowed to increase before aborting."""
        patience_const = patience
        validation_error = self.measure_error(self.validation_data, self.validation_data_solution)
        sample_count = len(self.input_layer)
        epoch_counter = 0
        for epoch in range(0, epochs*batch_size, batch_size):
            epoch_counter += 1
            self.back_propagate(self.input_layer[epoch%sample_count:(epoch%sample_count)+batch_size],
                                self.output_layer[epoch%sample_count:(epoch%sample_count)+batch_size], alpha, beta, momentum)
            validation_error_new = self.measure_error(self.validation_data, self.validation_data_solution)
            if  validation_error_new < validation_error:
                validation_error = validation_error_new
                patience = patience_const
                if auto_save:
                    self.save(auto_save) 
            else:
                patience -= 1
                if patience == 0:
                    print("Abort Training. Overfitting has started! Epoch: {0}. Error: {1}".format(epoch_counter, validation_error_new))
                    return
            if epoch % show == 0:
                print("Epoch: {0}, Validation Error: {1}".format(epoch_counter, validation_error))
                       
    
if __name__ == "__main__":
    #If the first row is a one the first output neuron should be on the second off
    x = np.array([  [0, 0, 1, 1, 0], 
                    [0, 1, 1, 1, 1], 
                    [1, 0, 1, 1, 1], 
                    [1, 1, 1, 1, 0], 
                    [0, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0] ])
    
    y = np.array([ [0, 1],
                  [0, 1],
                  [1, 0],
                  [1, 0],
                  [0, 1],
                  [1, 0],
                   [1, 0],
                   [1, 0] ])

    net = FeedForwardNetwork(input_dim=5, hidden_dim=200, output_dim=2)
    net.set_training_data(x, y)
    start = time.time()
    net.batch_train(epochs=2000, alpha=0.05, beta=0.0001, momentum=0.99, patience=20)
    print(time.time()-start)
