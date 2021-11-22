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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    pred = predictions.argmax(axis=1)
    accuracy = torch.true_divide((pred == targets).sum(), len(targets))

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model, data_loader, flag=False):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ac_list = []
    batch_sizes = []
    loss_ep = []
    loss_module = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.reshape(x.shape[0], -1)
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            ac_list.append(accuracy(pred,y))
            
            loss = loss_module(pred, y)
            loss_ep.append(loss.item())
            batch_sizes.append(len(y))
            
    avg_accuracy = np.average(ac_list, weights=batch_sizes)
    
    if flag == True:
        val_loss = np.average(loss_ep, weights=batch_sizes)
        return avg_accuracy, val_loss

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    train_loader = cifar10_loader['train']
    valid_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']

    # TODO: Initialize model and loss module
    model = MLP(3*32*32, hidden_dims, 10, use_batch_norm)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    model.to(device)
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    best_ac = -1
    model.train()
    for epoch in range(epochs):
        loss_ep = []
        batch_sizes = []
        for x, y in train_loader:
        
            x = x.reshape(x.shape[0], -1)
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            pred = pred.squeeze(dim=1)
            
            loss = loss_module(pred, y)
            loss_ep.append(loss.item())
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            batch_sizes.append(len(y))
            with torch.no_grad():
                train_loss = np.average(loss_ep, weights=batch_sizes)
            
        train_ac = evaluate_model(model, train_loader)
        train_accuracies.append(train_ac)
        train_losses.append(train_loss)
        #validate the model
        val_ac, val_loss = evaluate_model(model, valid_loader, flag = True)
        val_accuracies.append(val_ac)
        val_losses.append(val_loss)
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("train", epoch+1, epochs, train_loss, train_ac))
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("valid", epoch+1, epochs, val_loss, val_ac))
        print("-------------------------------------------------------------------------------------")
        if val_ac > best_ac:
            best_model = deepcopy(model)
            best_ac = val_ac
    # TODO: Test best model
    test_accuracy = evaluate_model(model, test_loader)
    print("Test_Accuracy: ", test_accuracy)
    # TODO: Add any information you might want to save for plotting
    logging_info = dict({"train_loss": train_losses, "train_ac": train_accuracies, "valid_loss": val_losses, "valid_ac": val_accuracies})
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info
    
    
def plot_results(list_train, list_valid, name):
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.plot(list_train, label="Training Set")
    plt.plot(list_valid, label="Validation Set")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    plot_results(logging_info['train_loss'], logging_info['valid_loss'], 'Loss')
    plot_results(logging_info['train_ac'], logging_info['valid_ac'], 'Accuracy')
    
