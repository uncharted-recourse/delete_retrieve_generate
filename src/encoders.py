import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_transfo_xl import RelPartialLearnableMultiHeadAttn, PositionalEmbedding
import copy
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

class MaskedRelPartialLearnableMultiHeadAttn(RelPartialLearnableMultiHeadAttn):
    """ subclass of RelPartialLearnableMultiHeadAttn that supports masking per instance"""
    def __init__(self, *args, **kwargs):
        super(MaskedRelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        
        #self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, attn_mask=None, key_mask = None, mems=None, head_mask=None):        
        # only post layer normalization
        
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        
        else:
            w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        r_head_k = self.r_net(r)
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + self.r_w_bias                                    # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = (attn_mask == 1)  # Switch to bool
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(
                        attn_mask[None,:,:,None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(
                        attn_mask[None,:,:,None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(
                        attn_mask[:,:,:,None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(
                        attn_mask[:,:,:,None], -1e30).type_as(attn_score)
        
        # mask by keys
        if key_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                attn_score = attn_score.masked_fill(
                    key_mask[None,:,:,None], -65000).type_as(attn_score)
            else:
                attn_score = attn_score.masked_fill(
                    key_mask[None,:,:,None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        ##### residual connection + layer normalization
        outputs = [self.layer_norm(w + attn_out)]

        if self.output_attentions:
            outputs.append(attn_prob)

        return outputs

class TransformerXLEncoderLayer(nn.TransformerEncoderLayer):
    r"""subclass of TransformerEncoderLayer 
        (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html)
        that optionally replaces MultiheadAttention with Relative MultiheadAttention. 

        rel_multihead_attention_partial_learnable: implementation from Transformer XL

        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_type = 'absolute'):
        super(TransformerXLEncoderLayer, self).__init__(d_model, nhead, dim_feedforward = dim_feedforward, dropout = dropout)
        
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
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)

    def forward(self, src, pos_emb, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional). 

        Shape:
            see the docs in Transformer class.
        """
        if self.attention_type == 'absolute':
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
        elif self.attention_type == 'relative':
            src = self.self_attn(src, pos_emb, attn_mask = src_mask, 
                                key_mask = src_key_padding_mask)[0]

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerXLEncoder(nn.Module):
    r"""TransformerXLEncoder is a stack of N TransformerXLEncoder layers

    Args:
        encoder_layer: an instance of the TransformerXLEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, d_model, num_layers, nhead = 8, dim_ff = 2048, dropout = 0.1, attention_type = 'absolute', clamp_len = 50):
        
        super(TransformerXLEncoder, self).__init__()
        self.attention_type = attention_type
        self.clamp_len = clamp_len
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        layer = TransformerXLEncoderLayer(d_model, nhead, dim_feedforward=dim_ff, attention_type=attention_type)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers

        # initialize weights
        self._reset_parameters()

    def forward(self, src, src_key_padding_mask=None):
        r"""Pass the input through the TransformerXLEncoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional). 

        """
        src = src.transpose(0,1)
        if self.attention_type == 'relative':
            src_key_padding_mask = src_key_padding_mask.transpose(0,1)
        qlen = src.shape[0]

        pos_seq = torch.arange(qlen-1, -1, -1.0, device=src.device, dtype=src.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        if self.attention_type == 'absolute':
            output = self.drop(src + pos_emb[-qlen:])
        elif self.attention_type == 'relative':
            output = self.drop(src)
            pos_emb = self.drop(pos_emb)

        for i in range(self.num_layers):
            output = self.layers[i](output, pos_emb, src_key_padding_mask=src_key_padding_mask)

        # layer norm
        output = self.norm(output)

        return output.transpose(0,1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)