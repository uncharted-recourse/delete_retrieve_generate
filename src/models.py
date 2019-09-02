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

def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None, map_location=None):
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

def initialize_inference_model(config=None):

    # read target data from training corpus to estalish attribute vocabulary / similarity
    log("reading training data from style corpus'", level="debug")
    
    src, tgt = data.read_nmt_data(
        src=config['data']['src'], 
        tgt=config['data']['tgt'],
        config=config,
        cache_dir=config['data']['vocab'],
        train_src = True,
        train_tgt=True
    )

    log("initializing model", level="debug")
    padding_id = data.get_padding_id(src['tokenizer'])
    model = SeqModel(
        src_vocab_size=len(src['tokenizer']),
        tgt_vocab_size=len(src['tokenizer']),
        pad_id_src=padding_id,
        pad_id_tgt=padding_id,
        config=config
    )
    if CUDA:
        model = model.cuda()

    return model, src, tgt

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
        elif self.options['encoder'] == 'transformer':
            # for now take default values of n_head
            self.encoder = encoders.TransformerEncoder(
                self.options['src_hidden_dim'],
                #dim_ff=self.options['src_hidden_dim'],
                dropout=self.options['dropout'],
                num_layers=self.options['src_layers']
            )
        self.ctx_bridge = nn.Linear(
            self.options['src_hidden_dim'],
            self.options['tgt_hidden_dim'])
        else:
            raise NotImplementedError('unknown encoder type')
        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.model_type == 'delete':
            if self.options['encoder'] == 'lstm':
                emb_dim = self.options['emb_dim']
            elif self.options['encoder'] == 'transformer':
                emb_dim = self.options['src_hidden_dim']
            self.attribute_embedding = nn.Embedding(
                # TODO change num to num styles supported
                num_embeddings=2, 
                embedding_dim=emb_dim)
            attr_size = emb_dim

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
                self.attribute_encoder = encoders.TransformerEncoder(
                    self.options['emb_dim'],
                    dim_ff=self.options['src_hidden_dim'],
                    dropout=self.options['dropout'],
                    num_layers=self.options['src_layers']
                )
                attr_size = self.options['src_hidden_dim']

        elif self.model_type == 'seq2seq':
            attr_size = 0

        else:
            raise NotImplementedError('unknown model type')

        self.c_bridge = nn.Linear(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])
        self.h_bridge = nn.Linear(
            attr_size + self.options['src_hidden_dim'], 
            self.options['tgt_hidden_dim'])

        # # # # # #  # # # # # #  # # # # # END NEW STUFF
        if self.options['decoder'] == 'lstm':
            self.decoder = decoders.StackedAttentionLSTM(config=config)
        elif self.options['decoder'] == 'transformer':
            self.decoder = decoders.TransformerDecoder(
                self.options['src_hidden_dim'],
                #dim_ff=self.options['src_hidden_dim'],
                dropout=self.options['dropout'],
                num_layers=self.options['tgt_layers']
            )
        else:
            raise NotImplementedError('unknown decoder type')

        # TODO: should we tie the weights of this output projection to input if embeddings are the same?
        self.output_projection = nn.Linear(
            self.options['tgt_hidden_dim'],
            tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

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

        srcmask = (1-srcmask).byte()

        if self.options['encoder'] == 'lstm':
            src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens, srcmask)
            
            if self.options['bidirectional']:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                c_t = src_c_t[-1]
            src_outputs = self.ctx_bridge(src_outputs)

        elif self.options['encoder'] == 'transformer':
            src_outputs = self.encoder(src_emb, srcmask)
            src_outputs = self.ctx_bridge(src_outputs)
            h_t = None
            c_t = None
        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)
            if self.options['encoder'] == 'lstm':
                c_t = torch.cat((c_t, a_ht), -1)
                c_t = self.c_bridge(c_t)
                h_t = torch.cat((h_t, a_ht), -1)
                h_t = self.h_bridge(h_t)
            elif self.options['encoder'] == 'transformer':
                a_ht = torch.unsqueeze(a_ht, 1)
                src_outputs = torch.cat((a_ht, src_outputs), 1)
                a_mask = Variable(torch.LongTensor([[0] for i in range(input_src.size(0))])).byte()
                if CUDA:
                    a_mask = a_mask.cuda()
                srcmask = torch.cat((a_mask, srcmask), dim = 1)
        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)

            if self.options['encoder'] == 'lstm':
                _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
                if self.options['bidirectional']:
                    a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                    a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
                else:
                    a_ht = a_ht[-1]
                    a_ct = a_ct[-1]
                c_t = torch.cat((c_t, a_ct), -1)
                c_t = self.c_bridge(c_t)
                h_t = torch.cat((h_t, a_ht), -1)
                h_t = self.h_bridge(h_t)

            elif self.options['encoder'] == 'transformer':
                a_ht = self.attribute_encoder(attr_emb, attrmask)
                src_outputs = torch.cat((a_ht, src_outputs), -1)


        # # # #  # # # #  # #  # # # # # # #  # # end diff
        tgt_emb = self.tgt_embedding(input_tgt)
        print(f'src size: {src_outputs.size()}')
        if self.options['decoder'] == 'lstm':
            tgt_outputs, (_, _) = self.decoder(
                tgt_emb,
                (h_t, c_t),
                src_outputs,
                srcmask)
        elif self.options['decoder'] == 'transformer':
            tgtmask = (1-tgtmask).byte()
            tgt_outputs = self.decoder(
                tgt_emb, 
                src_outputs, 
                tgtmask,
                srcmask)
        print(f'tgt output: {tgt_outputs.size()}')
        tgt_outputs_reshape = tgt_outputs.contiguous().view(
            tgt_outputs.size()[0] * tgt_outputs.size()[1],
            tgt_outputs.size()[2])
        print(f'tgt output reshape: {tgt_outputs_reshape.size()}')
        # Should we tie these weights to decoder input embedding?
        decoder_logit = self.output_projection(tgt_outputs_reshape)
        decoder_logit = decoder_logit.view(
            tgt_outputs.size()[0],
            tgt_outputs.size()[1],
            decoder_logit.size()[1])

        probs = self.softmax(decoder_logit)

        return decoder_logit, probs

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
        join_method = 'add',
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
        if self.join_method == "add":
            self.multp = nn.Parameter(torch.zeros(1))
            #self.multp = nn.Parameter(torch.rand(1))
        elif self.join_method == "gate":
            self.lm_sigmoid = nn.Sigmoid()
        else:
            raise Exception("join method must be 'gate' or 'add'")

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask, tgtmask):

        # generate predictions from language model
        lm_logit = self.language_model.forward(input_tgt)

        # generate s2s logits
        s2s_logit, _ = super(FusedSeqModel, self).forward(input_src,
            input_tgt,
            srcmask,
            srclens,
            input_attr,
            attrlens,
            attrmask,
            tgtmask
        )

        # add or multiply projected logits
        if self.join_method == "add":
            combined_logit = s2s_logit.add(lm_logit * self.multp.expand_as(lm_logit))
        elif self.join_method == 'gate':
            combined_logit = s2s_logit * self.lm_sigmoid(lm_logit)

        probs = self.softmax(combined_logit)

        return combined_logit, probs

    def count_params(self):
        return super(FusedSeqModel, self).count_params()


        

        
