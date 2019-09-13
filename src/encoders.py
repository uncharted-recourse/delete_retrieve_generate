import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.cuda import CUDA

class LSTMEncoder(nn.Module):
    """ simple wrapper for a bi-lstm """
    def __init__(self, emb_dim, hidden_dim, layers, bidirectional, dropout, pack=False):
        super(LSTMEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // self.num_directions,
            layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout)

        self.pack = pack

    def init_state(self, input):
        batch_size = input.size(0) # retrieve dynamically for decoding
        h0 = Variable(torch.zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.lstm.num_layers * self.num_directions,
            batch_size,
            self.lstm.hidden_size
        ), requires_grad=False)

        if CUDA:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0


    def forward(self, src_embedding, srclens):
        h0, c0 = self.init_state(src_embedding)

        if self.pack:
            inputs = pack_padded_sequence(src_embedding, srclens, batch_first=True)
        else:
            inputs = src_embedding

        outputs, (h_final, c_final) = self.lstm(inputs, (h0, c0))

        if self.pack:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs, (h_final, c_final)

class TransformerEncoder(nn.Module):
    """ simple wrapper for a pytorch transformer encoder """
    def __init__(self, emb_dim, n_head = 8, dim_ff = 1024, dropout = 0.1, num_layers = 4):
        r""" Encoder is a stack of N encoder layers"""
        super(TransformerEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            emb_dim, 
            n_head, 
            dim_feedforward = dim_ff, 
            dropout = dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers, 
            norm = nn.LayerNorm(emb_dim)
        )

    def forward(self, src_embedding, srcmask):
        r""" Pass the inputs (and masks) through each encoder layer in turn"""
        return self.transformer_encoder.forward(
            src_embedding.transpose(0,1), 
            src_key_padding_mask = srcmask
        ).transpose(0,1)
