###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set

from torch.utils.data import DataLoader
from copy import deepcopy

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
        cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model
    
def accuracy_fun(predictions, targets):
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


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir)
    test_dataset = get_test_set(data_dir)
    
    train_dataloader      = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataloader       = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                       
    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    
    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    model.to(device)

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    best_ac = -1
    
    for epoch in range(epochs):
        model.train()
        loss_ep = []
        ac_list = []
        batch_sizes = []
        for x, y in train_dataloader: 
            #sx = x.reshape(x.shape[0], -1)
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            loss = criterion(pred, y)
            loss_ep.append(loss.item())
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            batch_sizes.append(len(y))
        with torch.no_grad():
            train_loss = np.average(loss_ep, weights=batch_sizes)
                
        scheduler.step()
        train_ac = evaluate_model(model, train_dataloader, device)
        train_accuracies.append(train_ac)
        train_losses.append(train_loss)
       
        val_ac, val_loss = evaluate_model(model, validation_dataloader, device, True)
        val_accuracies.append(val_ac)
        val_losses.append(val_loss)
       
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("train", epoch+1, epochs, train_loss, train_ac))
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("valid", epoch+1, epochs, val_loss, val_ac))
        print("-------------------------------------------------------------------------------------")
       
        # Load best model and return it.
        if val_ac > best_ac:
            best_model = deepcopy(model)
            best_ac = val_ac
    torch.save(best_model, checkpoint_name)
    #######################
    # END OF YOUR CODE    #
    #######################
    return best_model


def evaluate_model(model, data_loader, device, flag=False):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    criterion = nn.CrossEntropyLoss()
    ac_list = []
    loss_ep = []
    batch_sizes = []
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            #x = x.reshape(x.shape[0], -1)
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            ac_list.append(accuracy_fun(pred,y).cpu())
            
            loss = criterion(pred, y)
            loss_ep.append(loss.item())
            batch_sizes.append(len(y))
            
    accuracy = np.average(ac_list, weights=batch_sizes)
    
    if flag == True:
        loss = np.average(loss_ep, weights=batch_sizes)
        return accuracy, loss
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model_name = str(model).split("(")[0]
    if model_name == "VGG":
        batch = str(model).split("(1): ")[1].split("(")[0]
        if batch == "BatchNorm2d":
    	    model_name = "VGG_bn"
    elif model_name == "ResNet":
        if len((str(model))) == 8147:
            model_name = "ResNet34"
        else:
            model_name = "ResNet18"
    elif model_name == "Sequential":
        model_name = "Clean"
    set_seed(seed)
    test_results = {}
    augmentation_list = [gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform]
    aug_list_name = ["gaussian_noise_transform", "gaussian_blur_transform", "contrast_transform", "jpeg_transform"]
    it = 0
    for augmentation in augmentation_list:
        for sev in range(5):
            test_dataset = get_test_set(data_dir, augmentation = augmentation(sev+1))
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            test_ac = evaluate_model(model, test_dataloader, device)
            test_results[aug_list_name[it]+" "+str(sev+1)] = test_ac
            print(aug_list_name[it]+" "+str(sev+1), test_ac)
        it += 1
    test_dataset = get_test_set(data_dir)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_ac = evaluate_model(model, test_dataloader, device)
    test_results["clean"] = test_ac
    print("None ", test_ac)
    with open(model_name+'_results.json', 'w') as fp:
        json.dump(test_results, fp)
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    set_seed(seed)
    
    checkpoint_name = "./"+model_name+".ckpt"
    model = get_model(model_name)
    best_model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device)
    test_model(best_model, batch_size, data_dir, device, seed)
    pass
    #######################
    # END OF YOUR CODE    #
    #######################





if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
