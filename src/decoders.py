import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.cuda import CUDA
from transformers.modeling_transfo_xl import PositionalEmbedding
from src.encoders import MaskedRelPartialLearnableMultiHeadAttn

class BilinearAttention(nn.Module):
    """ bilinear attention layer: score(H_j, q) = H_j^T W_a q
                (where W_a = self.in_projection)
    """
    def __init__(self, hidden):
        super(BilinearAttention, self).__init__()
        self.in_projection = nn.Linear(hidden, hidden, bias=False)
        self.softmax = nn.Softmax()
        self.out_projection = nn.Linear(hidden * 2, hidden, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, query, keys, srcmask=None, values=None):
        """
            query: [batch, hidden]
            keys: [batch, len, hidden]
            values: [batch, len, hidden] (optional, if none will = keys)

            compare query to keys, use the scores to do weighted sum of values
            if no value is specified, then values = keys
        """
        if values is None:
            values = keys
    
        # [Batch, Hidden, 1]
        decoder_hidden = self.in_projection(query).unsqueeze(2)
        # [Batch, Source length]
        attn_scores = torch.bmm(keys, decoder_hidden).squeeze(2)
        if srcmask is not None:
            attn_scores = attn_scores.masked_fill(srcmask, -float('inf'))
            
        attn_probs = self.softmax(attn_scores)
        # [Batch, 1, source length]
        attn_probs_transposed = attn_probs.unsqueeze(1)
        # [Batch, hidden]
        weighted_context = torch.bmm(attn_probs_transposed, values).squeeze(1)

        context_query_mixed = torch.cat((weighted_context, query), 1)
        context_query_mixed = self.tanh(self.out_projection(context_query_mixed))

        return weighted_context, context_query_mixed, attn_probs


class AttentionalLSTM(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_dim, hidden_dim, config, attention):
        """Initialize params."""
        super(AttentionalLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.use_attention = attention
        self.config = config
        self.cell = nn.LSTMCell(input_dim, hidden_dim)

        if self.use_attention:
            self.attention_layer = BilinearAttention(hidden_dim)


    def forward(self, input, hidden, ctx, srcmask, kb=None):
        input = input.transpose(0, 1)

        output = []
        timesteps = range(input.size(0))
        for i in timesteps:
            if type(hidden[0]) == type(None):
                hy, cy = self.cell(input[i])
            else:
                hy, cy = self.cell(input[i], hidden)
            if self.use_attention:
                _, h_tilde, alpha = self.attention_layer(hy, ctx, srcmask)
                hidden = h_tilde, cy
                output.append(h_tilde)
            else: 
                hidden = hy, cy
                output.append(hy)

        # combine outputs, and get into [time, batch, dim]
        output = torch.cat(output, 0).view(
            input.size(0), *output[0].size())
        output = output.transpose(0, 1)
        return output, hidden


class StackedAttentionLSTM(nn.Module):
    """ stacked lstm with input feeding
    """
    def __init__(self, cell_class=AttentionalLSTM, config=None):
        super(StackedAttentionLSTM, self).__init__()
        self.options=config['model']

        self.dropout = nn.Dropout(self.options['dropout'])

        self.layers = []
        input_dim = self.options['emb_dim']
        hidden_dim = self.options['tgt_hidden_dim']
        for i in range(self.options['tgt_layers']):
            layer = cell_class(input_dim, hidden_dim, config, config['model']['attention'])
            self.add_module('layer_%d' % i, layer)
            self.layers.append(layer)
            input_dim = hidden_dim


    def forward(self, input, hidden, ctx, srcmask, kb=None):
        h_final, c_final = [], []
        for i, layer in enumerate(self.layers):
            output, (h_final_i, c_final_i) = layer(input, hidden, ctx, srcmask, kb)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_final.append(h_final_i)
            c_final.append(c_final_i)

        h_final = torch.stack(h_final)
        c_final = torch.stack(c_final)

        return input, (h_final, c_final)

class TransformerXLDecoderLayer(nn.TransformerDecoderLayer):
    r"""subclass of TransformerDecoderLayer 
        (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html)
        that optionally replaces MultiheadAttention with Relative MultiheadAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_type = 'absolute'):
        super(TransformerXLDecoderLayer, self).__init__(d_model, nhead, dim_feedforward = dim_feedforward, dropout = dropout)

        self.attention_type = attention_type
        if attention_type == 'absolute':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif attention_type == 'relative':
            self.self_attn = MaskedRelPartialLearnableMultiHeadAttn(nhead, d_model, d_model // nhead, 
                dropout, dropatt = dropout)
            self.dropout1 = None
            self.norm1 = None
        

        else:
            raise NotImplementedError('attention_type must be "multihead_attention" or "rel_multihead_attention_partial_learnable"')

        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        # self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)

        # self.norm1 = LayerNorm(d_model)
        # self.norm2 = LayerNorm(d_model)
        # self.norm3 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)
        # self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, pos_emb, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if self.attention_type == 'absolute':
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        elif self.attention_type == 'relative':
            tgt = self.self_attn(tgt, pos_emb, attn_mask = tgt_mask, 
                                    key_mask = tgt_key_padding_mask)[0]
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerXLDecoder(nn.Module):
    r"""TransformerXLDecoder is a stack of N TransformerXLDecoder layers

    Args:
        encoder_layer: an instance of the TransformerXLDecoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, d_model, num_layers, nhead = 8, dim_ff = 2048, dropout = 0.1, attention_type = 'absolute', clamp_len = 50):

        super(TransformerXLDecoder, self).__init__()
        self.attention_type = attention_type
        self.clamp_len = clamp_len
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        layer = TransformerXLDecoderLayer(d_model, nhead, dim_feedforward=dim_ff, attention_type=attention_type)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers

        # initialize weights
        self._reset_parameters()

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the input through the TransformerXLDecoder layers in turn.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        """
        tgt = tgt.transpose(0,1)
        memory = memory.transpose(0,1)
        qlen = tgt.shape[0]

        if self.attention_type == 'absolute':
            pos_seq = torch.arange(0, max_len, device=tgt.device, dtype=tgt.dtype)
        elif self.attention_type == 'relative':
            pos_seq = torch.arange(qlen-1, -1, -1.0, device=tgt.device, dtype=tgt.dtype)
            tgt_key_padding_mask = tgt_key_padding_mask.transpose(0,1)

        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        if self.attention_type == 'absolute':
            output = self.drop(tgt + pos_emb[-qlen:])
            attn_mask = self.generate_square_subsequent_mask(qlen)
        elif self.attention_type == 'relative':
            output = self.drop(tgt)
            pos_emb = self.drop(pos_emb)
            attn_mask = torch.triu(torch.ones((qlen, qlen), dtype=torch.uint8), diagonal=1)[:,:,None]
        if CUDA:
            attn_mask = attn_mask.cuda()

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, pos_emb, tgt_mask=attn_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask = memory_key_padding_mask)

        output = self.norm(output)

        return output.transpose(0,1)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0). Copied from torch Transformer base class. 
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
