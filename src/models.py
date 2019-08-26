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

from src.cuda import CUDA
from src import data
from pytorch_transformers import OpenAIGPTModel, GPT2Model, XLNetModel, TransfoXLModel

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

    # overwrite tgt_dist_measurer for inference in both directions
    src['dist_measurer'] = data.CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt['content']],
            key_corpus=[' '.join(x) for x in src['content']],
            value_corpus=[' '.join(x) for x in src['attribute']],
            vectorizer=TfidfVectorizer(vocabulary=src['vocab']),
            make_binary=False
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
            self.ctx_bridge = nn.Linear(
                self.options['src_hidden_dim'],
                self.options['tgt_hidden_dim'])

        else:
            raise NotImplementedError('unknown encoder type')

        # # # # # #  # # # # # #  # # # # #  NEW STUFF FROM STD SEQ2SEQ
        
        if self.model_type == 'delete':
            self.attribute_embedding = nn.Embedding(
                num_embeddings=2, 
                embedding_dim=self.options['emb_dim'])
            attr_size = self.options['emb_dim']

        elif self.model_type == 'delete_retrieve':
            self.attribute_encoder = encoders.LSTMEncoder(
                self.options['emb_dim'],
                self.options['src_hidden_dim'],
                self.options['src_layers'],
                self.options['bidirectional'],
                self.options['dropout'],
                pack=False)
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

        self.decoder = decoders.StackedAttentionLSTM(config=config)

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

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask):
        src_emb = self.src_embedding(input_src)

        srcmask = (1-srcmask).byte()

        src_outputs, (src_h_t, src_c_t) = self.encoder(src_emb, srclens, srcmask)

        if self.options['bidirectional']:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        src_outputs = self.ctx_bridge(src_outputs)


        # # # #  # # # #  # #  # # # # # # #  # # seq2seq diff
        # join attribute with h/c then bridge 'em
        # TODO -- put this stuff in a method, overlaps w/above

        if self.model_type == 'delete':
            # just do h i guess?
            a_ht = self.attribute_embedding(input_attr)
            c_t = torch.cat((c_t, a_ht), -1)
            h_t = torch.cat((h_t, a_ht), -1)

        elif self.model_type == 'delete_retrieve':
            attr_emb = self.src_embedding(input_attr)
            _, (a_ht, a_ct) = self.attribute_encoder(attr_emb, attrlens, attrmask)
            if self.options['bidirectional']:
                a_ht = torch.cat((a_ht[-1], a_ht[-2]), 1)
                a_ct = torch.cat((a_ct[-1], a_ct[-2]), 1)
            else:
                a_ht = a_ht[-1]
                a_ct = a_ct[-1]

            h_t = torch.cat((h_t, a_ht), -1)
            c_t = torch.cat((c_t, a_ct), -1)
            
        c_t = self.c_bridge(c_t)
        h_t = self.h_bridge(h_t)

        # # # #  # # # #  # #  # # # # # # #  # # end diff

        tgt_emb = self.tgt_embedding(input_tgt)
        tgt_outputs, (_, _) = self.decoder(
            tgt_emb,
            (h_t, c_t),
            src_outputs,
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

        models = {
            'gpt': OpenAIGPTModel, 
            'gpt2': GPT2Model, 
            'xlnet': XLNetModel,
            'transformerxl': TransfoXLModel
        }
        model_weights = {
            'gpt': 'openai-gpt', 
            'gpt2': 'gpt2', 
            'xlnet': 'xlnet-base-cased',
            'transformerxl': 'transfo-xl-wt103'
        }

        model_name = self.config['data']['tokenizer']
        if model_name not in models.keys():
            raise Exception("Language model must be one of 'bert', 'gpt', 'gpt2', 'xlnet', 'transformerxl'")

        # !! assume that language model and seq2seq are using same tokenization !!
        self.language_model = models[model_name].from_pretrained(model_weights[model_name], 
            cache_dir=self.config['data']['working_dir']
        )

        # finetune if desired
        if CUDA:
            self.language_model = self.language_model.cuda()
        if not finetune:
            for param in self.language_model.parameters():
                param.requires_grad = False

        # define layers that join language model and sequence to sequence
        self.lm_output_projection = nn.Linear(
            self.language_model.config.hidden_size,
            self.tgt_vocab_size)

        # join language model and s2s model
        self.join_method = join_method
        if self.join_method == "add":
            self.multp = nn.Parameter(torch.rand(1))
        elif self.join_method == "gate":
            self.lm_sigmoid = nn.Sigmoid()
        else:
            raise Exception("join method must be 'gate' or 'add'")
        
        self.init_weights()

    def init_fused_weights(self):
        self.lm_output_projection.bias.data.fill_(0)

    def forward(self, input_src, input_tgt, srcmask, srclens, input_attr, attrlens, attrmask):

        # generate predictions from language model
        lm_features = self.language_model(input_src)[0]

        # project language model feature vector to vocabulary size
        lm_logit = self.lm_output_projection(lm_features)

        # generate s2s logits
        s2s_logit, _ = super(FusedSeqModel, self).forward(input_src,
            input_tgt,
            srcmask,
            srclens,
            input_attr,
            attrlens,
            attrmask)
        
        # add or multiply projected logits
        if self.join_method == "add":
            combined_logit = s2s_logit.add(lm_logit * self.multp)
        elif self.join_method == 'gate':
            combined_logit = s2s_logit * self.lm_sigmoid(lm_logit)

        probs = self.softmax(combined_logit)

        return combined_logit, probs

    def count_params(self):
        return super(FusedSeqModel, self).count_params()


        

        
