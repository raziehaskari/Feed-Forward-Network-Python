import numpy as np
from Network import FeedForwardNetwork

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

net = FeedForwardNetwork(input_dim=5, hidden_dim=5, output_dim=2)
net.set_training_data(x, y)
net.batch_train(epochs=200, alpha=0.01, beta=0.0001, momentum=0.99, patience=10)
