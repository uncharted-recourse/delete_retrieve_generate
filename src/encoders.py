import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
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
        #self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.shape[1]], 
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
        
        if CUDA:
            self.positions = self.positions.cuda()
        
        #self.register_buffer('positions', self.positions)
    
    def forward(self, x):
        x = x + self.embedding(self.positions[:, :x.shape[1]]).squeeze(1)
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
        y = y.transpose(2,3)
        z = z.transpose(1,2)
    xy_matmul = torch.matmul(x, y)
    # x_t_r is [length or 1, batch_size * heads, length or depth]
    # replace view with reshape if memory errors
    x_t_r = x.view(length, heads * batch_size, -1)
    # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z)
    # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = x_tz_matmul.view(batch_size, heads, length, -1)
    return xy_matmul + x_tz_matmul_r_t

def _generate_relative_positions_matrix(length, max_relative_position):
  """Generates matrix of relative positions between inputs.
     Original tensorflow implementation: https://github.com/tensorflow/tensor2tensor"""

    range_mat = torch.arange(length).repeat(length, 1)
    distance_mat = range_mat - range_mat.transpose(0,1)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, max_relative_position, embedding_layer):
  """Generates tensor of size [1 if cache else length, length, depth].
    Original tensorflow implementation: https://github.com/tensorflow/tensor2tensor"""

    relative_positions_matrix = _generate_relative_positions_matrix(length, max_relative_position)
    # Generates embedding for each relative position of dimension depth.
    return embedding_layer(relative_positions_matrix)

def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   k_embedding_layer, 
                                   v_embedding_layer,
                                   max_relative_position,
                                   dropout_rate=0.0):

  """Calculate relative position-aware dot-product self-attention.
  Original tensorflow implementation: https://github.com/tensorflow/tensor2tensor

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.
  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.

  Returns:
    A Tensor.
  Raises:
    ValueError: if max_relative_position is not > 0.
  """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attention." % (max_relative_position))

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    assert q.shape == k.shape, "This calculation only works for self attention. Q and K must have same shape"
    assert q.shape == v.shape, "This calculation only works for self attention. Q and K must have same shape"

    # Use separate embeddings suitable for keys and values.
    depth = k.shape[3]
    length = k.shape[2]
    relations_keys = _generate_relative_positions_embeddings(
        length, max_relative_position, k_embedding_layer)
    relations_values = _generate_relative_positions_embeddings(
        length, depth, max_relative_position, v_embedding_layer)

    # Compute self attention considering the relative position embeddings.

    ## Continue from here tomorrow - need separate impl. for masked relative pos. att. in decoder
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
        logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
        save_weights_to[scope.name] = weights
        save_weights_to[scope.name + "/logits"] = logits
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if not tf.get_variable_scope().reuse and make_image_summary:
        attention_image_summary(weights, image_shapes)
    return _relative_attention_inner(weights, v, relations_values, False)

class TransformerEncoder(nn.Module):
    """ simple wrapper for a pytorch transformer encoder """
    def __init__(self, emb_dim, n_head = 8, dim_ff = 1024, dropout = 0.1, num_layers = 4, max_len = 50, positional_encoding = 'embedding'):
        r""" Encoder is a stack of N encoder layers"""
        super(TransformerEncoder, self).__init__()

        # apply desired positional encoding
        if positional_encoding == 'embedding':
            self.pos_encoding = PositionalEmbedding(emb_dim, dropout=dropout)
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