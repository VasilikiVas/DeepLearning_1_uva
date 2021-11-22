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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import json
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.

def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    hidden_dim = [[128], [128, 256], [128, 256, 512]]
    use_batch_norm =[True, False]
    json_list = []
    for i in range(len(hidden_dim)):
        for j in range(len(use_batch_norm)):
            results = train_mlp_pytorch.train(hidden_dims=hidden_dim[i], epochs=20, use_batch_norm = use_batch_norm[j], lr=0.1, batch_size=128, seed=42, data_dir='data/')
            if use_batch_norm[j] == True:
                label = "With_batch_norm"
            else:
                label = "Without_batch_norm"
            results[3]['hidden_dim'] = str(hidden_dim[i])+"_"+label
            json_list.append(results[3])
    with open('results.json', 'w') as outfile:
        json.dump(json_list, outfile)
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, 'r') as j:
        contents = json.loads(j.read())
        
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Train Accuracy', fontsize=12)
    for i in range(len(contents)):
        plt.plot(contents[i]['train_ac'], label=str(contents[i]['hidden_dim']))
    plt.legend()
    plt.savefig('Train_ac.png')
    
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    for i in range(len(contents)):
        plt.plot(contents[i]['valid_ac'], label = str(contents[i]['hidden_dim']))
    plt.legend()
    plt.savefig('Valid_ac.png')
    
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    for i in range(len(contents)):
        plt.plot(contents[i]['train_loss'], label=str(contents[i]['hidden_dim']))
    plt.legend()
    plt.savefig('Train_Loss.png')
    
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Valid Loss', fontsize=12)
    for i in range(len(contents)):
        plt.plot(contents[i]['valid_loss'], label=str(contents[i]['hidden_dim']))
    plt.legend()
    plt.savefig('Valid_Loss.png')
    	
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.

    FILENAME = 'results.json' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)
