"""Sequence to Sequence models."""
import glob
import numpy as np
import os
import logging
from utils.log_func import get_log_func

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.decoders as decoders
import src.encoders as encoders
import src.discriminators as discriminators

from src.cuda import CUDA
from src import data

log_level = os.getenv("LOG_LEVEL", "WARNING")
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
log = get_log_func(__name__)

def get_latest_ckpt(ckpt_dir, model_type = 'model'):
    """ get latest model checkpoint from ckpt_dir"""    

    ckpts = glob.glob(os.path.join(ckpt_dir, model_type + '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None, map_location=None, model_type = 'model'):
    """ attempt to load model from directory or path and an (optional) device map location"""

    assert checkpoint_dir or checkpoint_path
    if checkpoint_dir:
        epoch, checkpoint_path = get_latest_ckpt(checkpoint_dir)
    else:
        epoch = int(checkpoint_path.split('.')[-2])

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        log('Load from %s sucessful!' % checkpoint_path, level="debug")
        return model, epoch + 1
    else:
        return model, 0

def initialize_inference_model(config, input_lines = None, map_location = 'cpu'):
    """ initialize inference model for deployment """

    # instantiate tokenizer (cached files copied to image)
    tokenizer = data.get_tokenizer(encoder = config['data']['tokenizer'], 
        cache_dir=os.path.join("checkpoints", config['data']['vocab_dir']))

    # init model
    vocab_size = len(tokenizer)
    model = FusedSeqModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        pad_id_src=tokenizer.pad_token_id,
        pad_id_tgt=tokenizer.pad_token_id,
        config=config
    )
    for param in model.parameters():
       param.requires_grad = False
    model.eval()

    # attempt to load model from working_dir in config
    model, _ = attempt_load_model(
        model=model,
        checkpoint_dir=f"checkpoints/{config['data']['working_dir']}",
        map_location=torch.device(map_location)
    )

    # if delete + retrieve paradigm, produce training data dictionaries for attribute
    # retrieval at inference time
    if input_lines is not None:
        train_data = data.read_nmt_data(input_lines, len(input_lines), config, train_data=None,
            cache_dir = config['data']['vocab_dir'])
    else:
        train_data = None

    return model, tokenizer, train_data

class SeqModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_id_src,
        pad_id_tgt,
        config=None,
    ):
        """Initialize model."""
        super(SeqModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt
        self.batch_size = config['data']['batch_size']
        self.config = config
        self.options = config['model']
        self.model_type = config['model']['model_type']

        self.src_embedding = nn.Embedding(
            self.src_vocab_size,
            self.options['emb_dim'],
            self.pad_id_src)

        if self.config['data']['share_vocab']:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                self.tgt_vocab_size,
                self.options['emb_dim'],
                self.pad_id_tgt)

        if self.options['encoder'] == 'lstm':
            self.encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'])
            ctx_bridge_in = self.options['src_hidden_dim']
            self.ctx_bridge = nn.Linear(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])
        elif self.options['encoder'] == 'transformer':
            # for now take default values of n_head
            self.encoder = encoders.TransformerXLEncoder(
                self.options['emb_dim'],
                self.options['src_layers'],
                dim_ff=self.options['src_hidden_dim'],
                dropout=self.options['dropout'],
                clamp_len=self.config['data']['max_len']
            )
            ctx_bridge_in = self.options['emb_dim']
        else:
            raise NotImplementedError('unknown encoder type')

        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        if self.model_type == 'delete':
            self.attribute_embedding = nn.Embedding(
                num_embeddings=len(config['data']['test']), 
                embedding_dim=self.options['emb_dim'])
            attr_size = self.options['emb_dim']

        elif self.model_type == 'delete_retrieve':
            if self.options['encoder'] == 'lstm':
                self.attribute_encoder = encoders.LSTMEncoder(
                    self.options['emb_dim'],
                    self.options['src_hidden_dim'],
                    self.options['src_layers'],
                    self.options['bidirectional'],
                    self.options['dropout'],
                    pack=False)
                attr_size = self.options['src_hidden_dim']
            elif self.options['encoder'] == 'transformer':
                # for now take default values of n_head
                self.attribute_encoder = encoders.TransformerXLEncoder(
                    self.options['emb_dim'],
                    self.options['src_layers'],
                    dim_ff=self.options['src_hidden_dim'],
                    dropout=self.options['dropout'],
                    clamp_len=self.config['data']['max_len'],
                )
                attr_size = self.options['emb_dim']

        elif self.model_type == 'seq2seq':
            attr_size = 0

        else:
            raise NotImplementedError('unknown model type')

        # # # # # #  # # # # # #  # # # # # END NEW STUFF
        if self.options['decoder'] == 'lstm':
            self.decoder = decoders.StackedAttentionLSTM(config=config)
            bridge_out = self.options['tgt_hidden_dim']
        elif self.options['decoder'] == 'transformer':
            self.decoder = decoders.TransformerXLDecoder(
                self.options['emb_dim'],
                self.options['tgt_layers'],
                dim_ff=self.options['tgt_hidden_dim'],
                dropout=self.options['dropout'],
                clamp_len=self.config['data']['max_len'],
            )
            bridge_out = self.options['emb_dim']
        else:
            raise NotImplementedError('unknown decoder type')
        
        if self.options['encoder'] == 'lstm':
            self.ctx_bridge = nn.Linear(ctx_bridge_in, bridge_out)
        elif self.options['encoder'] == 'transformer':
            self.ctx_bridge = nn.Linear(ctx_bridge_in + attr_size, bridge_out)
        self.c_bridge = nn.Linear(attr_size + ctx_bridge_in, bridge_out)
        self.h_bridge = nn.Linear(attr_size + ctx_bridge_in, bridge_out)

        self.output_projection = nn.Linear(
            bridge_out,
            tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()
        if self.options['tie_weights']:
            self.tie_weights(self.output_projection, self.tgt_embedding)

    def tie_weights(self, first_module, second_module):
        """Tie the input and output embeddings"""
        first_module.weight = second_module.weight
        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

    def init_weights(self):
        """Initialize weights.""" 
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.h_bridge.bias.data.fill_(0)
        self.c_bridge.bias.data.fill_(0)
        self.output_projection.bias.data.fill_(0)

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask, tgtmask):
        src_emb = self.src_embedding(input_src)

        if self.options['encoder'] == 'lstm':
            src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens)
            if self.options['bidirectional']:
                h_t_encoder = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t_encoder = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t_encoder = src_h_t[-1]
                c_t_encoder = src_c_t[-1]
            src_outputs = self.ctx_bridge(src_outputs)

        elif self.options['encoder'] == 'transformer':
            src_outputs_encoder = self.encoder(src_emb, srcmask)
            h_t_encoder = None
            c_t_encoder = None

        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        # attribute embedding can be average of different styles (indicator variables or
        # words) only utilized at inference time, not during training 
        if self.model_type == 'delete':
            a_hts = self.attribute_embedding(input_attr)
            a_ht = torch.mean(a_hts, dim=-2) if a_hts.shape[1] > 1 else a_hts.squeeze(1)
            if self.options['encoder'] == 'lstm':
                c_t = torch.cat((c_t_encoder, a_ht), -1)
                c_t = self.c_bridge(c_t)
                h_t = torch.cat((h_t_encoder, a_ht), -1)
                h_t = self.h_bridge(h_t)
            elif self.options['encoder'] == 'transformer':
                a_ht = torch.unsqueeze(a_ht, 1).expand_as(src_outputs_encoder)
                src_outputs = torch.cat((a_ht, src_outputs_encoder), -1)
                src_outputs = self.ctx_bridge(src_outputs)
                # a_mask = Variable(torch.BoolTensor([[False] for i in range(input_src.size(0))]))
                # if CUDA:
                #     a_mask = a_mask.cuda()
                # srcmask = torch.cat((a_mask, srcmask), dim = 1)
        elif self.model_type == 'delete_retrieve':
            attr_embs = self.src_embedding(input_attr)
            attr_emb = torch.mean(attr_embs, dim=-2) if attr_embs.shape[1] > 1 else attr_embs.squeeze(1)
            if self.options['encoder'] == 'lstm':
                _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens)
                if self.options['bidirectional']:
                    a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                    a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
                else:
                    a_ht = a_ht[-1]
                    a_ct = a_ct[-1]
                c_t = torch.cat((c_t_encoder, a_ct), -1)
                c_t = self.c_bridge(c_t)
                h_t = torch.cat((h_t_encoder, a_ht), -1)
                h_t = self.h_bridge(h_t)

            elif self.options['encoder'] == 'transformer':
                a_ht = self.attribute_encoder(attr_emb, attrmask)
                src_outputs = torch.cat((a_ht, src_outputs_encoder), -1)
                src_outputs = self.ctx_bridge(src_outputs)

        # # # #  # # # #  # #  # # # # # # #  # # end diff

        # multiply input_tgt by embedding weights if input is probability distribution
        if len(input_tgt.size()) == 3: # batch_size, seq_length, vocab_size
            tgt_emb = torch.matmul(input_tgt, self.tgt_embedding.weight)
        else:
            tgt_emb = self.tgt_embedding(input_tgt)

        if self.options['decoder'] == 'lstm':
            tgt_outputs, _ = self.decoder(
                tgt_emb,
                (h_t, c_t),
                src_outputs,
                srcmask)
        elif self.options['decoder'] == 'transformer':
            #tgtmask = (1-tgtmask).byte()
            tgt_outputs = self.decoder(
                tgt_emb, 
                src_outputs, 
                tgtmask,
                srcmask)

        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])

        decoder_logit = self.output_projection(tgt_outputs_reshape)
        decoder_logit = decoder_logit.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            decoder_logit.size()[1])

        probs = self.softmax(decoder_logit)

        # in adversarial paradigm, also return decoder output (equivalent to hidden states in lstm architecture)
        return decoder_logit, probs, tgt_outputs

    # returns trainable params, untrainable params
    def count_params(self):
        n_trainable_params = 0
        n_untrainable_params = 0
        for param in self.parameters():
            if param.requires_grad:
                n_trainable_params += np.prod(param.data.cpu().numpy().shape)
            else:
                n_untrainable_params += np.prod(param.data.cpu().numpy().shape)
        return n_trainable_params, n_untrainable_params

class FusedSeqModel(SeqModel):
    def __init__(
        self,
        *args,
        join_method = 'shallow',
        init_weights = 'zero',
        finetune = False,
        **kwargs,
    ):
        """Initialize model."""
        super(FusedSeqModel, self).__init__(*args, **kwargs)

        # initialize language model
        self.language_model = discriminators.LanguageModel(
            self.tgt_vocab_size,
            join_method=join_method,
            finetune=finetune,
            model_name=self.config['data']['tokenizer'],
            cache_dir=self.config['data']['lm_dir']
        )

        # join language model and s2s model
        self.join_method = join_method
        if self.join_method == "shallow":
            if init_weights == 'zero':
                self.multp = nn.Parameter(torch.zeros(1))
            elif init_weights == 'ones':
                self.multp = nn.Parameter(torch.ones(1))
            elif init_weights == 'rand':
                self.multp = nn.Parameter(torch.rand(1))
            else:
                raise NotImplementedError('init weights must be one of "zero", "ones", or "rand"')

        # deep fusion and cold fusion from https://arxiv.org/pdf/1708.06426.pdf
        # TODO: rewrite to operate on decoder hidden states and lm hidden states 
        elif self.join_method == "deep":
            lm_dim = self.language_model.lang_model.config.n_embd
            s2s_dim = self.options['emb_dim'] if self.options['decoder'] == 'transformer' else self.options['tgt_hidden_dim']
            self.lm_sigmoid = nn.Sigmoid()
            self.lm_relu = nn.ReLU()
            self.lm_linear_0 = nn.Linear(lm_dim, lm_dim)
            self.lm_linear_1 = nn.Linear(lm_dim + s2s_dim, self.tgt_vocab_size)
            self.init_fusion_weights()
        elif self.join_method == 'cold':
            lm_dim = self.language_model.lang_model.config.n_embd
            s2s_dim = self.options['emb_dim'] if self.options['decoder'] == 'transformer' else self.options['tgt_hidden_dim']
            self.lm_sigmoid = nn.Sigmoid()
            self.lm_relu = nn.ReLU()
            self.lm_linear_0 = nn.Linear(self.tgt_vocab_size, lm_dim)
            self.lm_linear_1 = nn.Linear(lm_dim + s2s_dim, lm_dim)
            self.lm_linear_2 = nn.Linear(lm_dim + s2s_dim, self.tgt_vocab_size)
            self.init_fusion_weights()
        else:
            raise NotImplementedError('join method must be "shallow", "deep" or "cold"')

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask, tgtmask):

        # generate predictions from language model
        lm_logit, lm_hidden_states = self.language_model.forward(input_tgt, attention_mask = ~tgtmask)

        # generate s2s logits
        s2s_logit, _, decoder_states = super(FusedSeqModel, self).forward(input_src,
            input_tgt,
            srcmask,
            srclens,
            input_attr,
            attrlens,
            attrmask,
            tgtmask
        )

        # combine lm logits and s2s logits according to fusion strategy
        if self.join_method == "shallow":
            combined = s2s_logit.add(lm_logit * self.multp.expand_as(lm_logit))
            probs = self.softmax(combined)
        elif self.join_method == 'deep':
            out = self.lm_linear_0(lm_hidden_states[-1])
            concat = torch.cat((decoder_states, self.lm_sigmoid(out)), 2)
            out = self.lm_linear_1(concat)
            combined = self.lm_relu(out)
            probs = self.softmax(combined)
        else:
            out = self.lm_linear_0(lm_logit)
            concat = torch.cat((decoder_states, out), 2)
            out2 = self.lm_linear_1(concat)
            gated = self.lm_sigmoid(out2)
            hadamard = gated * out
            concat = torch.cat((decoder_states, hadamard), 2)
            out = self.lm_linear_2(concat)
            combined = self.lm_relu(out)
            probs = self.softmax(combined)

        return combined, probs, decoder_states

    def count_params(self):
        return super(FusedSeqModel, self).count_params()

    def init_fusion_weights(self):
        """Initialize fusion weights.""" 
        for name, p in self.named_parameters():
            if name == 'lm_linear_0.weight' or name == 'lm_linear_1.weight':
                nn.init.xavier_uniform_(p)
            if self.join_method == 'cold':
                if name == 'lm_linear_2.weight':
                    nn.init.xavier_uniform_(p)



        

        
