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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ReLU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        super(MLP, self).__init__()
        self.linear_layers = nn.ModuleList()
        self.relu_functions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.n_hidden = n_hidden

        for i in range(len(n_hidden)+1):
            if i==0:
                self.linear_layers.append(nn.Linear(n_inputs, n_hidden[0]))
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(num_features=n_hidden[0]))
            elif i == len(n_hidden):
                self.linear_layers.append(nn.Linear(n_hidden[i-1], n_classes))
            else:
                self.linear_layers.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(num_features=n_hidden[i]))
            self.relu_functions.append(nn.ReLU())
        
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
            if i != len(self.n_hidden):
                if len(self.batch_norms) != 0:
                    out = self.batch_norms[i].forward(out)    
                out = self.relu_functions[i].forward(out)        

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
