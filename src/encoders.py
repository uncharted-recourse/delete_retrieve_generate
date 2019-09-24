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

class PositionalEncoding(nn.Module):
    "Implement the PE function. Copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention"

    def __init__(self, d_model, dropout = 0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    """ implements absolute position embeddings followed by dropout"""

    def __init__(self, d_model, dropout = 0.1, max_len=50):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(
                        max_len, 
                        d_model)

        # compute the absolute position encodings once
        self.positions = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('positions', self.positions)
    
    def forward(self, x):
        x = x + self.embedding(self.positions)
        return self.dropout(x)

def relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.
    Original tensorflow implementation: https://github.com/tensorflow/tensor2tensor

    This batches matrix multiply calculations to avoid unnecessary broadcasting.
    Args:
        x: Tensor with shape [batch_size, heads, length or 1, length or depth].
        y: Tensor with shape [batch_size, heads, length or 1, depth].
        z: Tensor with shape [length or 1, length, depth].
        transpose: Whether to transpose inner matrices of y and z. Should be true if
            last dimension of x is depth, not length.
    Returns:
        A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]

    # xy_matmul is [batch_size, heads, length or 1, length or depth]
    if transpose:
        y = transpose(2,3)
    xy_matmul = torch.matmul(x, y, transpose_b=transpose)
    # x_t is [length or 1, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    # x_t_r is [length or 1, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
    # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
    # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t

class TransformerEncoder(nn.Module):
    """ simple wrapper for a pytorch transformer encoder """
    def __init__(self, emb_dim, n_head = 8, dim_ff = 1024, dropout = 0.1, num_layers = 4, max_len = 50, positional_encoding = 'embedding'):
        r""" Encoder is a stack of N encoder layers"""
        super(TransformerEncoder, self).__init__()

        # apply desired positional encoding
        if positional_encoding == 'embedding':
            self.pos_encoding = PositionalEncoding(emb_dim, dropout=dropout)
        elif positional_encoding == 'sinusoid':
            self.pos_encoding = PositionalEncoding(emb_dim, dropout = dropout)
        else:
            raise NotImplementedError('positional encoding method must be "embedding" or "sinusoid"')

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

        # initialize weights
        self._reset_parameters()

    def forward(self, src_embedding, srcmask):
        r""" Pass the inputs (and masks) through each encoder layer in turn"""
        
        # run through positional encoding layer
        hidden_state = self.pos_encoding(src_embedding)

        return self.transformer_encoder.forward(
            hidden_state.transpose(0,1), 
            src_key_padding_mask = srcmask
        ).transpose(0,1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)