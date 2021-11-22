################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.linear_layers = []
        self.relu_functions = []
        self.n_hidden = n_hidden
        
        for i in range(len(n_hidden)+1):
            if i==0:
                self.linear_layers.append(LinearModule(n_inputs, n_hidden[0]))
            elif i == len(n_hidden):
                self.linear_layers.append(LinearModule(n_hidden[i-1], n_classes))
            else:
                self.linear_layers.append(LinearModule(n_hidden[i-1], n_hidden[i]))
            self.relu_functions.append(ReLUModule())
        self.softmax = SoftMaxModule()
        self.loss_function = CrossEntropyModule()

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x 
        for i in range(len(self.n_hidden)+1):
            out = self.linear_layers[i].forward(out)
            if i == len(self.n_hidden):
                out = self.softmax.forward(out)
            else:
                out = self.relu_functions[i].forward(out)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        for i in range(len(self.n_hidden), -1, -1):
            if i == len(self.n_hidden):
                dout = self.softmax.backward(dout)
            else:
                dout = self.relu_functions[i].backward(dout)
            dout = self.linear_layers[i].backward(dout)
        return

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
       
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.softmax = None
        self.loss_function = None
        self.linear_layers = None
        self.relu_functions = None
        self.n_hidden = None

        #######################
        # END OF YOUR CODE    #
        #######################
