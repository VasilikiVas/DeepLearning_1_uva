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

import math
import torch
import torch.nn as nn


from datetime import datetime
import argparse
from tqdm.auto import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextDataset, text_collate_fn



class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.W_gx = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim), requires_grad=True)
        self.W_ix = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim), requires_grad=True)
        self.W_fx = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim), requires_grad=True)
        self.W_ox = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim), requires_grad=True)
        
        self.W_gh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.W_ih = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.W_fh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.W_oh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True)
        self.W_ph = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=True)

        self.b_g = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        desired_bound = 1 / math.sqrt(self.hidden_dim)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter.data, -desired_bound, desired_bound)
        self.b_f.data += 1
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
        
        input_len, batch_size, _ = embeds.shape
        
        c_t = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_t = torch.zeros(batch_size, self.hidden_dim).to(device)

        it = 0
        for t in range(input_len):
            embed = embeds[t,:,:]
            g_t = torch.tanh(embed @ self.W_gx + h_t @ self.W_gh + self.b_g)
            i_t = torch.sigmoid(embed @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(embed @ self.W_fx + h_t @ self.W_fh + self.b_f)
            o_t = torch.sigmoid(embed @ self.W_ox + h_t @ self.W_oh + self.b_o)
            
            h_prev = h_t.unsqueeze(0)
            
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
            
            if it == 0:
                h_total = h_prev.detach().clone()
            else : 
                h_total = torch.cat((h_total, h_prev))
            it += 1
        
        #p_t = h_t @ self.W_ph + self.b_h
        #y_t = nn.Softmax(dim=1)(p_t)
        
        return h_total 
        #######################
        # END OF YOUR CODE    #
        #######################

'''if __name__ == "__main__":
    model = LSTM(1024, 256)
    txt_file = "assets/book_EN_democracy_in_the_US.txt"
    dataset = TextDataset(txt_file, 30)
    data_loader = DataLoader(dataset, 128, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)
    
    x, _ = next(iter(data_loader)) 
    embeddings = nn.Embedding(12508, 256)
    embed = embeddings(x)
    model.forward(embed)'''
    

class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.
        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.embedding_size = args.embedding_size
        self.vocabulary_size = args.vocabulary_size
        
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.lstm = LSTM(self.lstm_hidden_dim, self.embedding_size)
        self.linear = nn.Linear(self.lstm_hidden_dim, self.vocabulary_size)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        embed = self.embeddings(x)
        lstm_out = self.lstm(embed)
        out = self.linear(lstm_out)
        
        return out
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=2.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = torch.zeros(sample_length, batch_size, dtype=int)    

        with torch.no_grad():
            for t in range(sample_length):
                if t == 0:
                    x[t, :] = torch.randint(low=0, high=self.vocabulary_size, size = (1, batch_size), dtype=int)
                    #print(x.shape, " 0")
                else:
                    p_t = self.forward(x[t-1, :].unsqueeze(0))
                    p_t = p_t.squeeze(0)
                    if int(temperature) == 0:
                        x[t, :] = torch.argmax(p_t, dim=-1)
                        #print(x.shape, " no_temp")
                    else:
                        print(p_t.shape)
                        x[t, :] = torch.softmax(p_t / temperature, dim=0)
                        x[t, :] = torch.argmax(p_t, dim=-1)
                        #print(x.shape, " temp")
        return x
        
        #######################
        # END OF YOUR CODE    #
        #######################
