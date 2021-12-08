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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel
import matplotlib.pyplot as plt


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

    pred = predictions.argmax(axis=-1)
    accuracy = torch.true_divide((pred == targets).sum(), targets.size(0) * targets.size(1))

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy

def plot_results(accuracy, loss, args):

    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Train Accuracy', fontsize=12)
    plt.plot(accuracy, label="Training_Accuracy")
    plt.legend()
    plt.savefig('Train_ac_'+(str(args.txt_file).split("/")[1]).split(".")[0]+'.png')
    
    fig = plt.figure(figsize=(8, 6), dpi=70)
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.plot(loss, label="Training_Loss")
    plt.legend()
    plt.savefig('Train_Loss_'+(str(args.txt_file).split("/")[1]).split(".")[0]+'.png')


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    device = args.device
    
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.vocabulary_size = dataset.vocabulary_size
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)
    # Create model
    model = TextGenerationModel(args)
    # Create optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    
    # Training loop
    model.to(device)
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    best_ac = -1
    
    for epoch in range(args.num_epochs):
        model.train()
        loss_ep = []
        ac_list = []
        batch_sizes = []
        for x, y in data_loader:
            #sx = x.reshape(x.shape[0], -1)
            x = x.to(device)
            y = y.to(device)
            y_one_hot = torch.nn.functional.one_hot(y, num_classes = args.vocabulary_size).float()
            
            pred = model(x)
            #print(pred.shape, y_one_hot.shape)
            loss = criterion(pred.permute(0,2,1), y_one_hot.permute(0,2,1))
            loss_ep.append(loss.item())
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            batch_sizes.append(len(y))
            ac_list.append(accuracy_fun(pred,y).cpu())
            
        scheduler.step()
        with torch.no_grad():
            train_ac = np.average(ac_list, weights=batch_sizes)
            train_loss = np.average(loss_ep, weights=batch_sizes)
                
        train_accuracies.append(train_ac)
        train_losses.append(train_loss)
       
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("train", epoch+1, args.num_epochs, train_loss, train_ac))

        samples = model.sample(batch_size=1, sample_length=args.sample_length, temperature=0)
        print('Greedy Sampling             : ', dataset.convert_to_string(samples))

        samples = model.sample(batch_size=1, sample_length=args.sample_length, temperature=0.5)
        print('Random Sampling - temp = 0.5: ', dataset.convert_to_string(samples))

        samples = model.sample(batch_size=1, sample_length=args.sample_length, temperature=1.0)
        print('Random Sampling - temp = 1.0: ', dataset.convert_to_string(samples))

        samples = model.sample(batch_size=1, sample_length=args.sample_length, temperature=2.0)
        print('Random Sampling - temp = 2.0: ', dataset.convert_to_string(samples))

        print("-------------------------------------------------------------------------------------")
    torch.save(model, "./lstm_model_"+(str(args.txt_file).split("/")[1]).split(".")[0]+".ckpt")
    plot_results(train_accuracies, train_losses, args)
    
    return model
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--sample_length', type=int, default=30, help='Sample length for printing process')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    train(args)
