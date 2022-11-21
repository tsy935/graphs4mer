import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import sys
from model.decoders import SequenceDecoder



class LSTMModel(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        num_rnn_layers,
        output_dim,
        output_seq_len=1,
        temporal_pool="pool",
        dropout=0.0,
        add_decoder=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.output_dim = output_dim
        self.temporal_pool = temporal_pool
        self.add_decoder = add_decoder

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_rnn_layers, 
            batch_first=True,
            dropout=dropout,
        )

        # Linear decoder
        if add_decoder:
            self.decoder = SequenceDecoder(
                d_model=hidden_dim,
                d_output=output_dim,
                l_output=output_seq_len,
                use_lengths=False,
                mode=temporal_pool,
            )

    def forward(self, x):
        """
        Args:
            x: (batch_size, max_seq_len, input_dim)
        """
        batch_size, max_seq_len, input_dim = x.shape

        # project input_dim --> hidden_size
        x = self.fc1(x)  # (batch, seq_len, hidden_size)

        # initialize hidden states
        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size, x.device)

        # LSTM
        x, _ = self.lstm(
            x, (initial_hidden_state, initial_cell_state)
        )  # (batch_size, max_seq_len, rnn_units)

        # temporal pooling
        if self.add_decoder:
            x = self.decoder(x)
            if x.shape[1] == 1:
                x = x.squeeze(1)

        return x

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_rnn_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(device)
        )
        cell = (
            weight.new(self.num_rnn_layers, batch_size, self.hidden_dim)
            .zero_()
            .to(device)
        )
        return hidden, cell