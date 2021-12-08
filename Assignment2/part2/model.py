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

##REMOVE THE IMPORTS
from dataset import TextDataset, text_collate_fn

import argparse

from torch.utils.data import DataLoader
import numpy as np

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

        self.W_gx = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim))
        self.W_ix = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim))
        self.W_fx = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim))
        self.W_ox = nn.Parameter(torch.zeros(self.embed_dim, self.hidden_dim))
        
        self.W_gh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.W_ih = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.W_fh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.W_oh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.W_ph = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))

        self.b_g = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(1, self.hidden_dim))

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
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        device = self.W_fx.device
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
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

            h_prev = h_t.unsqueeze(0)

            if it == 0:
                h_total = h_prev.detach().clone()
            else :
                h_total = torch.cat((h_total, h_prev))
            it += 1

        return h_total

        #######################
        # END OF YOUR CODE    #
        #######################


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

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
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
        device = self.lstm.W_fx.device
        sampled_x = torch.zeros(sample_length, batch_size, dtype=int).to(device)
        
        with torch.no_grad():
            for t in range(sample_length):
                if t == 0:
                    sampled_x[t] = torch.randint(self.vocabulary_size, (1, batch_size))
                else:
                    p_t = self.forward(sampled_x[:t])
                    if temperature == 0:
                        sampled_x[t] = torch.argmax(p_t[t-1], dim=-1)
                    else:
                        next_chars = p_t[t-1] / temperature
                        p_w = torch.softmax(next_chars, dim=1)
                        sampled_x[t] = torch.multinomial(p_w, num_samples=1).squeeze(1)
        return sampled_x.squeeze_().tolist()

        #######################
        # END OF YOUR CODE    #
        #######################
